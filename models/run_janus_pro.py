import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Janus"))

from data_utils import get_dataset
from janus.models import MultiModalityCausalLM, VLChatProcessor
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Janus-Pro/janus_pro.jsonl")
parser.add_argument("--emotion", type=str, default="base_short")
parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument(
    "--yes_no",
    action="store_true",
    help="Whether to filter yes/no questions and force a Yes/No answer.",
)
args = parser.parse_args()

output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

if not torch.cuda.is_available() or not args.device.startswith("cuda"):
    raise RuntimeError(
        "Janus-Pro smoke test must run on CUDA. "
        "Check PyTorch/CUDA install and GPU visibility."
    )

vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": args.device},
)
vl_gpt = vl_gpt.eval()

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]

    prompt_text = render_prompt(args.emotion, question)
    if args.yes_no:
        prompt_text += (
            " Please answer with 'Yes' or 'No'. "
            "No extra explanation and filler words."
        )

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt_text}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True,
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    with torch.inference_mode():
        output_ids = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    decoded = tokenizer.decode(
        output_ids[0].cpu().tolist(),
        skip_special_tokens=True,
    ).strip()

    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
