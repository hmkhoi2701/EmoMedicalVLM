import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import get_dataset
from smoke_test.emotion_prompts import render_prompt
from utils import build_output_dict, load_image


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="output/HuatuoGPT-Vision/huatuogpt_vision.jsonl")
parser.add_argument("--emotion", type=str, default="default")
parser.add_argument("--model_path", type=str, default="FreedomIntelligence/HuatuoGPT-Vision-7B")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument(
    "--yes_no",
    action="store_true",
    help="Whether to filter yes/no questions and force a Yes/No answer.",
)
args = parser.parse_args()

huatuo_repo = Path(__file__).resolve().parent.parent / "HuatuoGPT-Vision"
if not huatuo_repo.exists():
    raise FileNotFoundError(
        f"HuatuoGPT-Vision repo not found at {huatuo_repo}. "
        "Clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git first."
    )

sys.path.insert(0, str(huatuo_repo))
from cli import HuatuoChatbot


output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

bot = HuatuoChatbot(args.model_path, device=args.device)
bot.gen_kwargs = {
    "do_sample": False,
    "max_new_tokens": args.max_new_tokens,
    "eos_token_id": bot.tokenizer.eos_token_id,
    "pad_token_id": bot.tokenizer.pad_token_id
    if bot.tokenizer.pad_token_id is not None
    else bot.tokenizer.eos_token_id,
}

samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]

    prompt_text = render_prompt(args.emotion, question)
    if args.yes_no:
        prompt_text += " Please answer with 'Yes' or 'No'. No extra explanation and filler words."

    decoded = bot.inference(prompt_text, [image])[0].strip()
    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
