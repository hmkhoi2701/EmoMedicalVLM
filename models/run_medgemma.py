from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import requests
from io import BytesIO
from copy import deepcopy
import torch
import json
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/medgemma.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and include them in the prompt.")
parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"], help="Whether to use single-turn or multi-turn conversation format.")
args = parser.parse_args()

if args.emotion == "default" and args.conv_mode == "multi":
    exit(0)

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

# Scoring
save_scores = args.yes_no and args.conv_mode == "single"
if save_scores:
    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("No")
    
# message    
if args.conv_mode == "single":
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": USER_PROMPTS[args.emotion]
                 + (" Question: {question}")
                 + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")},
            ],
        },
    ]
else:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPTS[args.emotion]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": """I'm here to help."""},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": " Question: {question}" + f"{'Please answer with Yes or No.' if args.yes_no else ''}"},
            ],
        },
    ]
    


# Run
for sample in tqdm(samples, desc="Processing samples"):
    if "SLAKE" in args.dataset:
        image = Image.open(sample["image"]).convert("RGB")
    elif "vqa-rad" in args.dataset:
        image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
    question = sample["question"]

    sample_messages = deepcopy(messages)
    sample_messages[-1]["content"][0]["image"] = image
    sample_messages[-1]["content"][1]["text"] = sample_messages[-1]["content"][1]["text"].format(question=question)
    
    inputs = processor.apply_chat_template(
        sample_messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    if save_scores:
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False, 
                                        output_scores=True, return_dict_in_generate=True)
        
        probs = generation['scores'][0][0].softmax(dim=-1)
        max_prob, max_id = probs.max(dim=-1)
        max_prob = round(max_prob.item(), 4)
        yes_prob = round(probs[yes_id].item(), 4)
        no_prob = round(probs[no_id].item(), 4)
        generated = generation['sequences'][0][input_len:]

        decoded = processor.decode(generated, skip_special_tokens=True).strip()
        
    else:    
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = processor.decode(generation, skip_special_tokens=True).strip()
    
    write_dict = {**sample, "model_answer": decoded, "dataset": args.dataset, "conv_mode": args.conv_mode}
    
    if save_scores:
        write_dict["max_prob"] = max_prob
        write_dict["yes_prob"] = yes_prob
        write_dict["no_prob"] = no_prob
        
    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
