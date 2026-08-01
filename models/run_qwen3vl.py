import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Qwen3-VL/qwen3vl.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
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

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

processor = AutoProcessor.from_pretrained(args.model_path)

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]

    prompt_text = render_prompt(args.emotion, question)
    if args.yes_no:
        prompt_text += " Please answer with 'Yes' or 'No'. No extra explanation and filler words."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    inputs = {
        key: value.to(torch.bfloat16)
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        else value
        for key, value in inputs.items()
    }

    input_ids = inputs["input_ids"]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = [
        output_ids[len(input_id):]
        for input_id, output_ids in zip(input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
