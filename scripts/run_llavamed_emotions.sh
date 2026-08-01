#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-microsoft/llava-med-v1.5-mistral-7b}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_SAMPLES=${MAX_SAMPLES:-}
YES_NO=${YES_NO:-1}
OUTPUT_DIR=${OUTPUT_DIR:-output/main_vqa/LLaVA-Med}

mapfile -t emotions < <("$PYTHON" - <<'PY'
from main_emotion_prompts import EMOTIONS

for emotion in EMOTIONS.values():
    for prompt in emotion["prompts"]:
        print(prompt["id"])
PY
)

datasets=("SLAKE" "vqa-med-2019")
datasets=("vqa-med-2019")
for dataset in "${datasets[@]}"
do
    dataset_slug=${dataset,,}
    dataset_slug=${dataset_slug//-/_}
    output_suffix=all
    if [ "$YES_NO" = "1" ]; then
        output_suffix=yes_no
    fi
    output_file="$OUTPUT_DIR/llavamed_${dataset_slug}_${output_suffix}.jsonl"

    for emotion in "${emotions[@]}"
    do
        args=(
            --model_path "$MODEL_PATH"
            --dataset "$dataset"
            --split test
            --emotion "$emotion"
            --prompt_set main
            --max_new_tokens "$MAX_NEW_TOKENS"
            --output_file "$output_file"
        )

        if [ "$YES_NO" = "1" ]; then
            args+=(--yes_no)
        fi
        if [ -n "$MAX_SAMPLES" ]; then
            args+=(--max_samples "$MAX_SAMPLES")
        fi

        echo "Running LLaVA-Med: dataset=$dataset prompt=$emotion yes_no=$YES_NO"
        "$PYTHON" models/run_llavamed.py "${args[@]}"
    done
done
