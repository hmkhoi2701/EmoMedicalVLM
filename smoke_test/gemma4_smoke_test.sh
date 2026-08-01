#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-google/gemma-4-E4B-it}
OUTPUT_FILE=${OUTPUT_FILE:-output/smoke_test/gemma4_e4b.jsonl}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

"$PYTHON" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "Gemma 4 requires Python >=3.10 for the current Transformers source "
        f"implementation. Current Python: {sys.version.split()[0]}"
    )

try:
    import transformers
except ImportError as exc:
    raise SystemExit(f"Could not import Transformers: {exc}")

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"Could not import PyTorch: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "Gemma 4 smoke test must run on CUDA, but torch.cuda.is_available() "
        "is False. Check the PyTorch CUDA wheel and NVIDIA driver."
    )

has_gemma4 = (
    hasattr(transformers, "AutoModelForMultimodalLM")
    or hasattr(transformers, "Gemma4ForConditionalGeneration")
)
if not has_gemma4:
    raise SystemExit(
        "Gemma 4 support is missing from Transformers "
        f"{transformers.__version__}. Install source Transformers in this "
        "Python >=3.10 env:\n"
        "python -m pip install git+https://github.com/huggingface/transformers.git"
    )
PY

for emotion in $("$PYTHON" -c 'from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS; print(" ".join(list(SMOKE_TEST_CONTROLS) + [p["prompt_id"] for e in SMOKE_TEST_EMOTIONS.values() for p in e["templates"]]))')
do
    echo "Testing emotion: $emotion with yes/no flag enabled"
    "$PYTHON" models/run_gemma4.py \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        --dataset "vqa-rad" \
        --split "test" \
        --emotion "$emotion" \
        --yes_no \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_file "$OUTPUT_FILE"
done
