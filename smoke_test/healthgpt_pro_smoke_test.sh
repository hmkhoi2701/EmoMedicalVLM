#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-lintw/HealthGPT-Pro-8B}
OUTPUT_FILE=${OUTPUT_FILE:-output/smoke_test/healthgpt_pro_8b.jsonl}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

"$PYTHON" - <<'PY'
try:
    import torch
except ImportError as exc:
    raise SystemExit(f"Could not import PyTorch: {exc}")

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:
    raise SystemExit(f"Could not import Qwen3VLForConditionalGeneration: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "HealthGPT-Pro smoke test must run on CUDA, but torch.cuda.is_available() "
        "is False. Check the PyTorch CUDA wheel and NVIDIA driver."
    )
PY

for emotion in $("$PYTHON" -c 'from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS; print(" ".join(list(SMOKE_TEST_CONTROLS) + [p["prompt_id"] for e in SMOKE_TEST_EMOTIONS.values() for p in e["templates"]]))')
do
    echo "Testing emotion: $emotion with yes/no flag enabled"
    "$PYTHON" models/run_healthgpt_pro.py \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        --dataset "vqa-rad" \
        --split "test" \
        --emotion "$emotion" \
        --yes_no \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_file "$OUTPUT_FILE"
done
