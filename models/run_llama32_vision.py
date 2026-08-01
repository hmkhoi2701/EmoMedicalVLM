import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Llama-3.2-Vision/llama32_vision_instruct.jsonl")
parser.add_argument("--emotion", type=str, default="base_short")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
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
        "Llama 3.2 Vision smoke test must run on CUDA. "
        "Check PyTorch/CUDA install and GPU visibility."
    )

model = MllamaForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    device_map={"": args.device},
).eval()
model.generation_config.temperature = None
model.generation_config.top_p = None

processor = AutoProcessor.from_pretrained(args.model_path)

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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        images=image,
        text=prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    inputs = {
        key: value.to(torch.bfloat16)
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        else value
        for key, value in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    decoded = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
