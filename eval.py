# import transformers
# import torch

# import json
# import os
# from tqdm import tqdm

# import argparse

# from transformers import pipeline
# import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen2.5-7B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="cuda"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# parser = argparse.ArgumentParser()
# parser.add_argument("--output_dir", type=str, default="output", help="The directory to save results.")
# parser.add_argument("--file_prefix", type=str, default="medgemma_", help="Run eval on a specific file prefix")
# args = parser.parse_args()

# messages = [
#     {"role": "system", "content": "You are an annotator for medical Q&A System.\
#         You will be given a question with a ground truth answer, and a LLM response.\
#         Your task is to evaluate the correctness of the LLM response.\
#         Your evaluation should be one of the following:\n\
#         1. Correct: The LLM response is correct and matches the ground truth answer.\n\
#         2. Incorrect: The LLM response is incorrect and does not match the ground truth answer.\n\
#         3. Partially Correct: The LLM response is partially correct. It contains some correct information but also has some inaccuracies or missing details compared to the ground truth answer.\n\
#         4. Irrelevant: The LLM response is irrelevant to the question and does not address the topic of the ground truth answer.\n\
#         For Yes/No questions, the evaluation should be Correct if the LLM response correctly answers Yes or No, and Incorrect otherwise.\n\
#         Reason your answer carefully but concisely, and provide your final evaluation at the end.\
#         Please provide your evaluation solely as: Evaluation: <Correct/Incorrect/Partially Correct/Irrelevant>."},
#     {"role": "user", "content": "Question: {question}\n\
#                                 Ground Truth Answer: {ground_truth}.\n\
#                                 LLM Response: {llm_response}."},
# ]

# for file in os.listdir(args.output_dir):
#     if file.startswith(args.file_prefix) and file.endswith(".jsonl"):
#         print(f"Evaluating {file}...")
#         with open(os.path.join(args.output_dir, file), "r") as f:
#             data = [json.loads(line) for line in f]
        
#             for item in tqdm(data):
#                 question = item["question"]
#                 ground_truth = item["answer"]
#                 llm_response = item["model_answer"]

#                 input_messages = [
#                     {"role": "system", "content": messages[0]["content"]},
#                     {"role": "user", "content": messages[1]["content"].format(
#                         question=question,
#                         ground_truth=ground_truth,
#                         llm_response=llm_response,
#                     )},
#                 ]
                
#                 text = tokenizer.apply_chat_template(
#                     input_messages,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
                
#                 model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#                 generated_ids = model.generate(
#                     **model_inputs,
#                     max_new_tokens=512,
#                     do_sample=False,
#                 )
#                 generated_ids = [
#                     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#                 ]

#                 response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#                 evaluation = response.strip().split("Evaluation:")[-1].strip()
#                 item["evaluation"] = evaluation
#         with open(os.path.join(args.output_dir, "eval_"+file), "w") as f:
#             for item in data:
#                 f.write(json.dumps(item) + "\n")
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

# để batch được an toàn hơn
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="The directory to save results.")
parser.add_argument("--file_prefix", type=str, default="medgemma_", help="Run eval on a specific file prefix")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
args = parser.parse_args()

messages = [
    {"role": "system", "content": "You are an annotator for medical Q&A System.\
        You will be given a question with a ground truth answer, and a LLM response.\
        Your task is to evaluate the correctness of the LLM response.\
        Your evaluation should be one of the following:\n\
        1. Correct: The LLM response is correct and matches the ground truth answer.\n\
        2. Incorrect: The LLM response is incorrect and does not match the ground truth answer.\n\
        3. Partially Correct: The LLM response is partially correct. It contains some correct information but also has some inaccuracies or missing details compared to the ground truth answer.\n\
        4. Irrelevant: The LLM response is irrelevant to the question and does not address the topic of the ground truth answer.\n\
        For Yes/No questions, the evaluation should be Correct if the LLM response correctly answers Yes or No, and Incorrect otherwise.\n\
        Reason your answer carefully but concisely, and provide your final evaluation at the end.\n\
        After reasoning, format evaluation solely as: Evaluation: <Correct/Incorrect/Partially Correct/Irrelevant>. NO OTHER TEXT."},
    {"role": "user", "content": "Question: {question}\n\n\
                                Ground Truth Answer: {ground_truth}.\n\
                                LLM Response: {llm_response}."},
]

for file in sorted(os.listdir(args.output_dir)):
    if file.startswith(args.file_prefix) and file.endswith(".jsonl"):
        print(f"Evaluating {file}...")
        with open(os.path.join(args.output_dir, file), "r") as f:
            data = [json.loads(line) for line in f]

        for start in tqdm(range(0, len(data), args.batch_size)):
            batch = data[start:start + args.batch_size]

            texts = []
            for item in batch:
                question = item["question"]
                ground_truth = item["answer"]
                llm_response = item["model_answer"]

                input_messages = [
                    {"role": "system", "content": messages[0]["content"]},
                    {"role": "user", "content": messages[1]["content"].format(
                        question=question,
                        ground_truth=ground_truth,
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
                if "Partially Correct" in evaluation:
                    evaluation = "Partially Correct"
                elif "Correct" in evaluation:
                    evaluation = "Correct"
                elif "Incorrect" in evaluation:
                    evaluation = "Incorrect"
                elif "Irrelevant" in evaluation:
                    evaluation = "Irrelevant"
                else:
                    pass
                item["evaluation"] = evaluation

        with open(os.path.join(args.output_dir, "eval_" + file), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")