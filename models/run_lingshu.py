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
from utils import build_output_dict, load_image

print("THIS IS NEW VERSION WITH MAX_SAMPLES")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS

USER_PROMPTS_MAIN = {
    **{name: prompt["template"] for name, prompt in SMOKE_TEST_CONTROLS.items()},
    **{
        prompt["prompt_id"]: prompt["template"]
        for emotion in SMOKE_TEST_EMOTIONS.values()
        for prompt in emotion["templates"]
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/lingshu.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and include them in the prompt.")
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

samples = get_dataset(args.dataset, args.split, args.yes_no)

if args.max_samples is not None:
    samples = samples[:args.max_samples]
    print(f"Running only first {len(samples)} samples.")

for sample in tqdm(samples, desc="Processing"):
    try:
        image = load_image(sample, args.dataset.lower())
        question = sample["question"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": ("Use the provided image to answer the user's question, ideally as a short phrase."
                                                                    "Do not assume additional patient history or findings that are not visually supported by the image or explicitly stated in the question.")}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPTS_MAIN[args.emotion].format(question=question) + (" Please answer with 'Yes' or 'No'. No extra explanation and filler words." if args.yes_no else "")},
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
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids = outputs[0][input_len:]
        decoded = processor.decode(generated_ids, skip_special_tokens=True).strip()

        write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

        with open(args.output_file, "a") as f:
            f.write(json.dumps(write_dict) + "\n")

    except Exception as e:
        print(f"Inference failed: {e}")

print("Done.")
# ===== END =====
