import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_TILES = 12


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=MAX_TILES, image_size=INPUT_SIZE, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def preprocess_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image)}")

    transform = build_transform(INPUT_SIZE)
    images = dynamic_preprocess(image, image_size=INPUT_SIZE, use_thumbnail=True, max_num=MAX_TILES)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/InternVL3_5/internvl35.jsonl")
parser.add_argument("--emotion", type=str, default="base_short")
parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B-Instruct")
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
        "InternVL3.5 smoke test must run on CUDA. "
        "Check PyTorch/CUDA install and GPU visibility."
    )

model = AutoModel.from_pretrained(
    args.model_path,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().to(args.device)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_fast=False,
)

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

generation_config = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": False,
}

for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]

    prompt_text = render_prompt(args.emotion, question)
    if args.yes_no:
        prompt_text += (
            " Please answer with 'Yes' or 'No'. "
            "No extra explanation and filler words."
        )

    pixel_values = preprocess_image(image).to(dtype=torch.bfloat16, device=args.device)
    response = model.chat(
        tokenizer,
        pixel_values,
        f"<image>\n{prompt_text}",
        generation_config,
    ).strip()

    write_dict = build_output_dict(sample, question, response, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
