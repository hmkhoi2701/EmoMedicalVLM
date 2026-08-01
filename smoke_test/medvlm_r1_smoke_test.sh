#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-JZPeterPan/MedVLM-R1}
OUTPUT_FILE=${OUTPUT_FILE:-output/smoke_test/medvlm_r1.jsonl}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

for emotion in $("$PYTHON" -c 'from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS; print(" ".join(list(SMOKE_TEST_CONTROLS) + [p["prompt_id"] for e in SMOKE_TEST_EMOTIONS.values() for p in e["templates"]]))')
do
    echo "Testing emotion: $emotion with yes/no flag enabled"
    "$PYTHON" models/run_medvlm_r1.py \
        --model_path "$MODEL_PATH" \
        --dataset "vqa-rad" \
        --split "test" \
        --emotion "$emotion" \
        --yes_no \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_file "$OUTPUT_FILE"
done
