import argparse
import torch
import json
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LLaVA-Med"))

from data_utils import get_dataset
from emotion_prompts import USER_PROMPTS

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/llavamed.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b", help="LLaVA-Med model path.")
parser.add_argument("--model_base", type=str, default=None, help="Optional base model path.")
parser.add_argument("--llava_conv_mode", type=str, default="mistral_instruct", help="LLaVA conversation template mode.")
parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"], help="Whether to use single-turn or multi-turn conversation format.")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling.")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams.")
parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum generated tokens.")
parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of samples to process.")
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and force a Yes/No answer.")
args = parser.parse_args()

if args.emotion == "default" and args.conv_mode == "multi":
    exit(0)

output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

disable_torch_init()
model_name = Path(args.model_path).name
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path,
    args.model_base,
    model_name,
)

device = next(model.parameters()).device

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

save_scores = args.yes_no and args.conv_mode == "single"
if save_scores:
    yes_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Yes"))[0]
    no_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("No"))[0]

if args.conv_mode == "single":
    prompt_template = (
        USER_PROMPTS[args.emotion]
        + " Question: {question}"
        + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")
    )
else:
    prompt_template = " Question: {question}" + (" Please answer with 'Yes' or 'No'." if args.yes_no else "")

for sample in tqdm(samples, desc="Processing samples"):
    if "SLAKE" in args.dataset:
        image = Image.open(sample["image"]).convert("RGB")
    elif "vqa-rad" in args.dataset:
        image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    question = sample["question"]

    conv = conv_templates[args.llava_conv_mode].copy()

    if args.conv_mode == "single":
        qs = prompt_template.format(question=question).strip()

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

    else:
        first_user = USER_PROMPTS[args.emotion].strip()
        second_user = prompt_template.format(question=question).strip()

        conv.append_message(conv.roles[0], first_user)
        conv.append_message(conv.roles[1], "I'm here to help.")

        if getattr(model.config, "mm_use_im_start_end", False):
            second_user = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + second_user
        else:
            second_user = DEFAULT_IMAGE_TOKEN + "\n" + second_user

        conv.append_message(conv.roles[0], second_user)
        conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device=device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)

    if save_scores:
        with torch.inference_mode():
            generation = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        probs = generation.scores[0][0].softmax(dim=-1)

        token_max_prob, _ = probs.max(dim=-1)
        token_max_prob = round(token_max_prob.item(), 4)

        yes_raw = probs[yes_id].item()
        no_raw = probs[no_id].item()

        binary_total = yes_raw + no_raw
        if binary_total > 0:
            yes_prob = round(yes_raw / binary_total, 4)
            no_prob = round(no_raw / binary_total, 4)
        else:
            yes_prob = 0.0
            no_prob = 0.0

        max_prob = max(yes_prob, no_prob)

        decoded = tokenizer.decode(generation.sequences[0], skip_special_tokens=True).strip()

        decoded_lower = decoded.lower().strip()
        if decoded_lower.startswith("yes"):
            decoded = "Yes"
        elif decoded_lower.startswith("no"):
            decoded = "No"

    else:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if args.yes_no:
            decoded_lower = decoded.lower().strip()
            if decoded_lower.startswith("yes"):
                decoded = "Yes"
            elif decoded_lower.startswith("no"):
                decoded = "No"

    write_dict = {
        "question": sample["question"],
        "answer": sample["answer"],
        "answer_type": "CLOSED" if args.yes_no else "OPEN",
        "model_answer": decoded,
        "dataset": args.dataset,
        "conv_mode": args.conv_mode,
    }

    if save_scores:
        write_dict["max_prob"] = max_prob
        write_dict["yes_prob"] = yes_prob
        write_dict["no_prob"] = no_prob

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")