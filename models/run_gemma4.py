import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/Gemma-4/gemma4.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--model_path", type=str, default="google/gemma-4-E4B-it")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument(
    "--yes_no",
    action="store_true",
    help="Whether to filter yes/no questions and force a Yes/No answer.",
)
args = parser.parse_args()


def get_transformers_components():
    try:
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "google/gemma-4-E4B-it needs a Transformers build with Gemma 4 "
            "support. Current Python is "
            f"{sys.version.split()[0]}; source Transformers now requires "
            "Python >=3.10."
        ) from exc

    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_class is None:
        model_class = getattr(transformers, "Gemma4ForConditionalGeneration", None)

    if model_class is None:
        raise ImportError(
            "google/gemma-4-E4B-it requires Gemma 4 support in Transformers. "
            f"Current env has Python {sys.version.split()[0]} and "
            f"Transformers {transformers.__version__}. Use a Python >=3.10 env "
            "and install Transformers from source: "
            "python -m pip install git+https://github.com/huggingface/transformers.git"
        )

    return AutoProcessor, model_class


AutoProcessor, Gemma4ModelClass = get_transformers_components()


def parse_gemma_response(processor, response):
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("answer", "response", "content", "text"):
                if key in parsed and parsed[key] is not None:
                    return str(parsed[key]).strip()
        return str(parsed).strip()

    return response.strip()


output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

if not torch.cuda.is_available() or not args.device.startswith("cuda"):
    raise RuntimeError("Gemma 4 smoke test must run on CUDA. Check PyTorch/CUDA install and GPU visibility.")

processor = AutoProcessor.from_pretrained(args.model_path)
model = Gemma4ModelClass.from_pretrained(
    args.model_path,
    dtype="auto",
    device_map={"": args.device},
).eval()

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

    chat_kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        inputs = processor.apply_chat_template(
            messages,
            enable_thinking=False,
            **chat_kwargs,
        )
    except TypeError:
        inputs = processor.apply_chat_template(messages, **chat_kwargs)

    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    response = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=False,
    )
    decoded = parse_gemma_response(processor, response)

    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
