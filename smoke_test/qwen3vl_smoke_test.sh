#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
OUTPUT_FILE=${OUTPUT_FILE:-output/smoke_test/qwen3vl_4b.jsonl}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

for emotion in $("$PYTHON" -c 'from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS; print(" ".join(list(SMOKE_TEST_CONTROLS) + [p["prompt_id"] for e in SMOKE_TEST_EMOTIONS.values() for p in e["templates"]]))')
do
    echo "Testing emotion: $emotion with yes/no flag enabled"
    "$PYTHON" models/run_qwen3vl.py \
        --model_path "$MODEL_PATH" \
        --dataset "vqa-rad" \
        --split "test" \
        --emotion "$emotion" \
        --yes_no \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_file "$OUTPUT_FILE"
done
