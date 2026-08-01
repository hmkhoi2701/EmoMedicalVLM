#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python}
DATASET=${DATASET:-vqa-rad}
SPLIT=${SPLIT:-test}
OUTPUT_FILE=${OUTPUT_FILE:-output/smoke_test/hulumed.jsonl}
TMP_DIR=${TMP_DIR:-output/smoke_test/hulumed_resume_tmp}
RUNNER=${RUNNER:-models/run_hulumed.py}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
DRY_RUN=${DRY_RUN:-0}

export DATASET SPLIT OUTPUT_FILE

mkdir -p "$TMP_DIR"

expected_count=$("$PYTHON" - <<'PY'
import os
from data_utils import get_dataset

print(len(get_dataset(os.environ["DATASET"], os.environ["SPLIT"], yes_no=True)))
PY
)

mapfile -t emotions < <("$PYTHON" - <<'PY'
from smoke_test.emotion_prompts import SMOKE_TEST_CONTROLS, SMOKE_TEST_EMOTIONS

prompt_ids = list(SMOKE_TEST_CONTROLS)
prompt_ids.extend(
    prompt["prompt_id"]
    for emotion_group in SMOKE_TEST_EMOTIONS.values()
    for prompt in emotion_group["templates"]
)

for prompt_id in prompt_ids:
    print(prompt_id)
PY
)

echo "Expected rows per prompt: $expected_count"
echo "Output file: $OUTPUT_FILE"

for emotion in "${emotions[@]}"
do
    export EMOTION="$emotion"

    existing_count=$("$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output_file = Path(os.environ["OUTPUT_FILE"])
emotion = os.environ["EMOTION"]
seen = set()

if output_file.exists():
    with output_file.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("emotion") != emotion:
                continue

            key = (
                str(row.get("image", "")),
                str(row.get("question", "")),
                str(row.get("answer", "")),
            )
            seen.add(key)

print(len(seen))
PY
)

    if [ "$existing_count" -ge "$expected_count" ]; then
        echo "Skipping $emotion: $existing_count/$expected_count complete"
        continue
    fi

    tmp_file="$TMP_DIR/${emotion}.jsonl"
    rm -f "$tmp_file"

    echo "Running $emotion: $existing_count/$expected_count complete"

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN=1, not running model for $emotion"
        continue
    fi

    run_status=0
    "$PYTHON" "$RUNNER" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --emotion "$emotion" \
        --yes_no \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_file "$tmp_file" || run_status=$?

    if [ -s "$tmp_file" ]; then
        export TMP_FILE="$tmp_file"
        added_count=$("$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output_file = Path(os.environ["OUTPUT_FILE"])
tmp_file = Path(os.environ["TMP_FILE"])
emotion = os.environ["EMOTION"]

def key_for(row):
    return (
        str(row.get("image", "")),
        str(row.get("question", "")),
        str(row.get("answer", "")),
    )

seen = set()
if output_file.exists():
    with output_file.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("emotion") == emotion:
                seen.add(key_for(row))

to_add = []
tmp_seen = set()
with tmp_file.open() as f:
    for line in f:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if row.get("emotion") != emotion:
            continue

        key = key_for(row)
        if key in seen or key in tmp_seen:
            continue

        to_add.append(row)
        tmp_seen.add(key)

with output_file.open("a") as f:
    for row in to_add:
        f.write(json.dumps(row) + "\n")

print(len(to_add))
PY
)
        echo "Merged $added_count new rows for $emotion"
    else
        echo "No temp output written for $emotion"
    fi

    if [ "$run_status" -ne 0 ]; then
        echo "Run failed for $emotion with status $run_status after merging partial temp output"
        exit "$run_status"
    fi
done

echo "Resume pass complete."
