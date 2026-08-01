import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, get_image_path

from transformers import AutoModelForCausalLM, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Hulu-Med/hulumed.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--model_path", type=str, default="ZJU-AI4H/Hulu-Med-4B")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and force a Yes/No answer.")
args = parser.parse_args()

output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
)

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

for sample in tqdm(samples, desc="Processing samples"):
    image_path = get_image_path(sample, args.dataset.lower(), output_dir / "images")
    prompt_text = render_prompt(args.emotion, sample["question"])
    if args.yes_no:
        prompt_text += " Please answer with 'Yes' or 'No'. No extra explanation and filler words."

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "image_path": image_path,
                    },
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": False,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        use_think=False,
    )[0].strip()

    write_dict = build_output_dict(sample, sample["question"], decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
