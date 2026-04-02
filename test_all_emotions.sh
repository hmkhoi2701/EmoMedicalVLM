#!/bin/bash

for emotion in \
"main_patient_anger_frustration" \
"main_patient_sadness_distress" \
"main_clinician_neutral" \
"main_clinician_fear_anxiety" \
"main_clinician_anger_frustration" \
"main_clinician_sadness_distress"
do
    echo "Running emotion: $emotion"

    python models/run_lingshu.py \
        --dataset "SLAKE" \
        --split "test" \
        --emotion "$emotion" \
        --output_file "output/lingshu_${emotion}.jsonl"
done
