# ===== COPY BELOW =====
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
import sys

print("THIS IS NEW VERSION WITH MAX_SAMPLES")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS_MAIN

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/lingshu.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--max_samples", type=int, default=None)
args = parser.parse_args()

os.makedirs(Path(args.output_file).parent, exist_ok=True)

model_name = "lingshu-medical-mllm/Lingshu-I-8B"

print("Loading model...")

processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="cuda",
    dtype=torch.float16,
).eval()

print("Model loaded.")

samples = get_dataset(args.dataset, args.split)

if args.max_samples is not None:
    samples = samples[:args.max_samples]
    print(f"Running only first {len(samples)} samples.")

for sample in tqdm(samples, desc="Processing"):
    try:
        image = Image.open(sample["image"]).convert("RGB")
        question = sample["question"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPTS_MAIN[args.emotion].format(question=question)},
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids = outputs[0][input_len:]
        decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()

        write_dict = {
            "image": sample["image"],
            "question": question,
            "answer": sample.get("answer", ""),
            "model_answer": decoded,
            "emotion": args.emotion,
        }

        with open(args.output_file, "a") as f:
            f.write(json.dumps(write_dict) + "\n")

    except Exception as e:
        print(f"Inference failed: {e}")

print("Done.")
# ===== END =====
