#!/bin/bash
set -euo pipefail

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

for emotion in $(python -c 'from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS; print(" ".join(list(SMOKE_TEST_CONTROLS) + [p["prompt_id"] for e in SMOKE_TEST_EMOTIONS.values() for p in e["templates"]]))')
do
    echo "Testing emotion: $emotion with yes/no flag enabled"
    python models/run_lingshu.py \
        --dataset "vqa-rad" \
        --split "test" \
        --emotion "$emotion" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --yes_no \
        --output_file "output/smoke_test/lingshu.jsonl"
done
