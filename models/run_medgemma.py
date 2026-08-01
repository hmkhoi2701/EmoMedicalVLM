from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import requests
import torch
import json
import os
from tqdm import tqdm
from utils import build_output_dict, load_image

import sys
from pathlib import Path
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
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/medgemma.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generated tokens.")
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and include them in the prompt.")
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

# Load model and processor
model_id = "google/medgemma-1.5-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

# Load dataset
samples = get_dataset(args.dataset, args.split, args.yes_no)

# Run
for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": ("Use the provided image to answer the user's question, ideally as a short phrase."
                                        "Do not assume additional patient history or findings that are not visually supported by the image or explicitly stated in the question.")}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPTS_MAIN[args.emotion].format(question=question) + (" Please answer with 'Yes' or 'No'. No extra explanation and filler words." if args.yes_no else "")},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True).strip()
    
    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)
    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
