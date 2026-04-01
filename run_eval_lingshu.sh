#!/bin/bash
set -e

python eval.py \
  --output_dir output \
  --file_prefix lingshu_ \
  --batch_size 4 \
  --judge_model Qwen/Qwen2.5-7B-Instruct