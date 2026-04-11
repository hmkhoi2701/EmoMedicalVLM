import transformers
import torch

import json
import os
from tqdm import tqdm

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="The directory to save results.")
parser.add_argument("--file_prefix", type=str, default="medgemma_", help="Run eval on a specific file prefix")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
args = parser.parse_args()

messages = [
    {
        "role": "system",
        "content": (
            "You are an annotator for a medical Q&A system. "
            "You will be given a question and an LLM response. "
            "Your task is to determine whether the LLM's response implies 'Yes' or 'No'. "
            "Respond strictly in the format: Evaluation: <Yes/No>. "
            "Do not include any additional text, explanation, or punctuation."
        )
    },
    {
        "role": "user",
        "content": (
            "Question: {question}\n\n"
            "LLM Response: {llm_response}"
        )
    },
]

for file in sorted(os.listdir(args.output_dir)):
    if file.startswith(args.file_prefix) and file.endswith(".jsonl") and "_closed" in file:
        print(f"Evaluating {file}...")
        with open(os.path.join(args.output_dir, file), "r") as f:
            data = [json.loads(line) for line in f]

        for start in tqdm(range(0, len(data), args.batch_size)):
            batch = data[start:start + args.batch_size]

            texts = []
            for item in batch:
                question = item["question"]
                llm_response = item["model_answer"]

                input_messages = [
                    {"role": "system", "content": messages[0]["content"]},
                    {"role": "user", "content": messages[1]["content"].format(
                        question=question,
                        llm_response=llm_response,
                    )},
                ]

                text = tokenizer.apply_chat_template(
                    input_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)

            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for item, response in zip(batch, responses):
                evaluation = response.strip().split("Evaluation:")[-1].strip()
                if evaluation.startswith("Yes"):
                    evaluation = "Yes"
                elif evaluation.startswith("No"):
                    evaluation = "No"
                else:
                    evaluation = "Unknown"
                item["evaluation"] = evaluation

        with open(os.path.join(args.output_dir, "eval_" + file), "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")