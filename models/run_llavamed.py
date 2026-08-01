import argparse
import torch
import json
import os
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LLaVA-Med"))

from data_utils import get_dataset
from main_emotion_prompts import EMOTIONS
from smoke_test.emotion_prompts import render_prompt as render_smoke_prompt
from utils import build_output_dict, load_image

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


PROMPTS_BY_ID = {
    prompt["id"]: prompt
    for emotion in EMOTIONS.values()
    for prompt in emotion["prompts"]
}


def render_main_prompt(prompt_id, question):
    emotion_context = PROMPTS_BY_ID[prompt_id]["prompt"]
    return f"{emotion_context}\n\nQuestion: {question}"


def render_prompt(prompt_id, question, prompt_set):
    if prompt_set == "main":
        if prompt_id not in PROMPTS_BY_ID:
            raise ValueError(f"Unknown main emotion prompt: {prompt_id}")
        return render_main_prompt(prompt_id, question)

    if prompt_set == "smoke":
        try:
            return render_smoke_prompt(prompt_id, question)
        except KeyError as error:
            raise ValueError(f"Unknown smoke-test prompt: {prompt_id}") from error

    if prompt_id in PROMPTS_BY_ID:
        return render_main_prompt(prompt_id, question)
    try:
        return render_smoke_prompt(prompt_id, question)
    except KeyError as error:
        raise ValueError(f"Unknown emotion prompt: {prompt_id}") from error


def get_answer_token_id(tokenizer, answer):
    for text in (answer, f" {answer}"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            return token_ids[0]
    raise ValueError(f"'{answer}' is not a single token for this tokenizer.")


def get_yes_no_scores(first_token_logits, first_generated_token_id, tokenizer):
    log_probs = torch.log_softmax(first_token_logits.float(), dim=-1)
    probs = log_probs.exp()

    yes_token_id = get_answer_token_id(tokenizer, "Yes")
    no_token_id = get_answer_token_id(tokenizer, "No")
    binary_probs = torch.softmax(
        first_token_logits.float()[[yes_token_id, no_token_id]], dim=-1
    )

    return {
        "generated_token": tokenizer.decode([first_generated_token_id]),
        "token_probability": log_probs[first_generated_token_id].exp().item(),
        "token_entropy": torch.special.entr(probs).sum().item(),
        "yes_probability": binary_probs[0].item(),
        "no_probability": binary_probs[1].item(),
        "binary_entropy": torch.special.entr(binary_probs).sum().item(),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
parser.add_argument("--output_file", type=str, default="output/llavamed.jsonl", help="The file to save the model's predictions to.")
parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
parser.add_argument(
    "--prompt_set",
    choices=["auto", "main", "smoke"],
    default="auto",
    help="Prompt definitions to use. 'auto' detects the set from the prompt ID.",
)
parser.add_argument("--model_path", type=str, default="microsoft/llava-med-v1.5-mistral-7b", help="LLaVA-Med model path.")
parser.add_argument("--model_base", type=str, default=None, help="Optional base model path.")
parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation template mode.")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generated tokens.")
parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of samples to process.")
parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and force a Yes/No answer.")
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

# Load model
disable_torch_init()
model_name = Path(args.model_path).name
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path,
    args.model_base,
    model_name,
)

device = next(model.parameters()).device

# Load dataset
samples = get_dataset(args.dataset, args.split, args.yes_no)
if args.max_samples is not None:
    samples = samples[:args.max_samples]

# Run
for sample in tqdm(samples, desc="Processing samples"):
    image = load_image(sample, args.dataset.lower())
    question = sample["question"]

    user_prompt = render_prompt(args.emotion, question, args.prompt_set)
    if args.yes_no:
        user_prompt += " Please answer with 'Yes' or 'No'. No extra explanation and filler words."
    qs = user_prompt.strip()

    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
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


    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if args.yes_no:
        generate_kwargs.update(
            return_dict_in_generate=True,
            output_scores=True,
        )

    with torch.inference_mode():
        generation_output = model.generate(
            input_ids,
            images=image_tensor,
            **generate_kwargs,
        )

    if args.yes_no:
        output_ids = generation_output.sequences
    else:
        output_ids = generation_output
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    write_dict = build_output_dict(sample, question, decoded, args.emotion, args.dataset)
    if args.yes_no:
        if not generation_output.scores:
            raise RuntimeError("Generation returned no token scores for yes/no output.")
        first_generated_token_id = output_ids[0, -len(generation_output.scores)].item()
        write_dict.update(
            get_yes_no_scores(
                generation_output.scores[0][0],
                first_generated_token_id,
                tokenizer,
            )
        )

    with open(args.output_file, "a") as f:
        f.write(json.dumps(write_dict) + "\n")
