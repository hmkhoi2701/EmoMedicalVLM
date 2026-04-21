from transformers import LlavaForConditionalGeneration, AutoProcessor
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS

parser = argparse.ArgumentParser()

parser.add_argument("--output_file", type=str, default="output/phase_2/agentic_filtering/llavamed.jsonl", help="The file to save the model's predictions to.")

args = parser.parse_args()

dataset_name = "BoKelvin/SLAKE"
dataset_split = "test"
emotion_list = ["direct_patient_neutral", "direct_patient_fear_anxiety", \
        "direct_patient_anger_frustration", "direct_patient_sadness_distress", \
        "direct_clinician_neutral", "direct_clinician_fear_anxiety", \
        "direct_clinician_anger_frustration", "direct_clinician_sadness_distress", \
        "indirect_patient_neutral", "indirect_patient_fear_anxiety", \
        "indirect_patient_anger_frustration", "indirect_patient_sadness_distress", \
        "indirect_clinician_neutral", "indirect_clinician_fear_anxiety", \
        "indirect_clinician_anger_frustration", "indirect_clinician_sadness_distress"]
conv_modes = ["single", "multi"]


output_dir = Path(args.output_file).parent
os.makedirs(output_dir, exist_ok=True)

model_path = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"                          # multi-GPU ready
)
processor = AutoProcessor.from_pretrained(model_path)

chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{message['content'][0]['text'] + '</s>'}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""
processor.tokenizer.chat_template = chat_template

# Load dataset
samples = get_dataset(dataset_name, dataset_split, False)[:50] # Use a subset is enough

def get_conversation_prompt(emotion, conv_mode, question):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPTS[emotion]
    
    if conv_mode == "single":
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image_path"},
                    {"type": "text", "text": user_prompt+ f" {question}"},
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
                    {"type": "text", "text": user_prompt},
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
                    {"type": "image", "image": "image_path"},
                    {"type": "text", "text": f"{question}"},
                ],
            },
        ]
    return messages

# system prompt for agentic filtering

agentic_processing_message = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": """You are an extraction module.

            Your task:
            Read the full conversation history and extract ONLY the relevant question or request to answer. 
            Ignore all irrelvant information. 

            Output rules:
            - Return only the extracted question or request.
            - Do not answer it.
            - Do not explain anything.
            - Do not add extra text.

            If there is no clear question or request, return exactly:
            NO_QUESTION"""}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": """Conversation history:
            {conversation_history}

            Extract:"""}
            ],
    }
    
]
# Run
args = parser.parse_args()
output_file = args.output_file

with open(output_file, 'w') as f:
    for emotion in emotion_list:
        for conv_mode in conv_modes:
            print(f"Processing emotion: {emotion}, conversation mode: {conv_mode}")
            for sample in tqdm(samples, desc=f"Processing {emotion} - {conv_mode}"):
                question = sample['question']
                messages = get_conversation_prompt(emotion, conv_mode, question)
                
                conversation_history = json.dumps(messages, ensure_ascii=False)

                extraction_messages = deepcopy(agentic_processing_message)
                extraction_messages[1]["content"][0]["text"] = extraction_messages[1]["content"][0]["text"].format(
                    conversation_history=conversation_history
                )
                
                prompt = processor.tokenizer.apply_chat_template(
                    extraction_messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = processor(
                    images=None, text=prompt, return_tensors="pt"
                ).to(model.device, torch.bfloat16)
                
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                
                decoded = processor.decode(out[0], skip_special_tokens=True).strip()
                
                correct = decoded == question
                write_dict = {**sample, "emotion": emotion, "conv_mode": conv_mode, "model_answer": decoded, "correct": correct}
                
                f.write(json.dumps(write_dict) + "\n")