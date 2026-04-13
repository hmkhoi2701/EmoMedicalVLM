#!/bin/bash
set -u

SCRIPT_PATH="models/run_medgemma.py"
OUTPUT_DIR="output/phase_2/MedGemma"
LOG_DIR="logs/medgemma"
RUN_ALL_LOG="${LOG_DIR}/run_all.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

touch "$RUN_ALL_LOG"

emotions=(
  "default"
  "direct_patient_neutral"
  "direct_patient_fear_anxiety"
  "direct_patient_anger_frustration"
  "direct_patient_sadness_distress"
  "direct_clinician_neutral"
  "direct_clinician_fear_anxiety"
  "direct_clinician_anger_frustration"
  "direct_clinician_sadness_distress"
  "indirect_patient_neutral"
  "indirect_patient_fear_anxiety"
  "indirect_patient_anger_frustration"
  "indirect_patient_sadness_distress"
  "indirect_clinician_neutral"
  "indirect_clinician_fear_anxiety"
  "indirect_clinician_anger_frustration"
  "indirect_clinician_sadness_distress"
)

datasets=("SLAKE" "vqa-rad")
conv_modes=("single" "multi")

log_msg () {
  local msg="$1"
  echo "$msg"
  echo "$msg" >> "$RUN_ALL_LOG"
}

run_job () {
  local dataset="$1"
  local conv_mode="$2"
  local emotion="$3"
  local yesno_flag="$4"

  local suffix=""
  local yesno_arg=""
  local job_type="open"
  local batch_size=2
  local max_new_tokens=128

  if [ "$yesno_flag" = "closed" ]; then
    suffix="_closed"
    yesno_arg="--yes_no"
    job_type="closed"
    batch_size=4
    max_new_tokens=8
  fi

  local output_file="${OUTPUT_DIR}/medgemma_${dataset}_${conv_mode}_${emotion}${suffix}.jsonl"

  log_msg "============================================================"
  log_msg "[RUN ] dataset=$dataset conv_mode=$conv_mode emotion=$emotion type=$job_type"
  log_msg "[OUT ] $output_file"
  log_msg "[CFG ] batch_size=$batch_size max_new_tokens=$max_new_tokens"

  # ❗关键：不再重定向到单独 log
  # python 自己会写 logs/medgemma/*.log
  python "$SCRIPT_PATH" \
    --dataset "$dataset" \
    --split "test" \
    --emotion "$emotion" \
    --conv_mode "$conv_mode" \
    --output_file "$output_file" \
    --batch_size "$batch_size" \
    --max_new_tokens "$max_new_tokens" \
    $yesno_arg

  local exit_code=$?

  if [ $exit_code -ne 0 ]; then
    log_msg "[FAIL] dataset=$dataset conv_mode=$conv_mode emotion=$emotion type=$job_type"
    return $exit_code
  fi

  if [ -f "$output_file" ] && [ -s "$output_file" ]; then
    log_msg "[DONE] $output_file"
  else
    log_msg "[WARN] output empty: $output_file"
  fi

  return 0
}

log_msg "🚀 Starting MedGemma batch run"
log_msg "SCRIPT_PATH=$SCRIPT_PATH"
log_msg "OUTPUT_DIR=$OUTPUT_DIR"

for emotion in "${emotions[@]}"; do
  for dataset in "${datasets[@]}"; do
    for conv_mode in "${conv_modes[@]}"; do

      if [ "$emotion" = "default" ] && [ "$conv_mode" = "multi" ]; then
        log_msg "[SKIP] default + multi"
        continue
      fi

      run_job "$dataset" "$conv_mode" "$emotion" "open"
      run_job "$dataset" "$conv_mode" "$emotion" "closed"

    done
  done
done

log_msg "✅ All jobs processed."