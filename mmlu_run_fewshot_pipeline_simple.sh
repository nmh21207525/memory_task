#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-1.5B-Instruct"
DATASET="mmlu"
GPU_IDS="2,3"
TASK_NAMES=""
EVAL_SPLIT="test"
FEW_SHOT_K=5
TRAIN_SIZE=""
EPOCHS=3
REPEAT_TIMES=10
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKSPACE_DIR="/mnt/niumiaohe/pwd_memory_task/workspace_fewshot"
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
SKIP_TRAINING=false

usage() {
  cat <<EOF
Usage:
  bash mmlu_run_fewshot_pipeline_simple.sh --model_path /path/to/model [options]

Required:
  --model_path PATH

Common options:
  --gpu_ids 0,1,2,3
  --task_names abstract_algebra,anatomy   MMLU subject list, empty means all subjects
  --few_shot_k 5
  --train_size 5                          default = few_shot_k
  --epochs 3
  --repeat_times 10
  --results_dir PATH
  --workspace_dir PATH
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 4
  --skip_training true|false

Split semantics:
  MMLU: for each subject, only test split is used in full flow.
        The first train_size samples are used for TTT/SFT and few-shot support;
        the remaining samples in that same test split are evaluated.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --task_names) TASK_NAMES="$2"; shift 2 ;;
    --few_shot_k) FEW_SHOT_K="$2"; shift 2 ;;
    --train_size) TRAIN_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --repeat_times) REPEAT_TIMES="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --workspace_dir) WORKSPACE_DIR="$2"; shift 2 ;;
    --per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --skip_training) SKIP_TRAINING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  usage
  exit 1
fi

if [[ -z "$TRAIN_SIZE" ]]; then
  TRAIN_SIZE="$FEW_SHOT_K"
fi

echo "Run few-shot pipeline"
echo "  model_path  = ${MODEL_PATH}"
echo "  dataset     = ${DATASET}"
echo "  gpu_ids     = ${GPU_IDS}"
echo "  task_names  = ${TASK_NAMES:-<all subjects>}"
echo "  eval_split  = test (fixed for mmlu)"
echo "  few_shot_k  = ${FEW_SHOT_K}"
echo "  train_size  = ${TRAIN_SIZE}"
echo "  epochs      = ${EPOCHS}"
echo "  repeat_times = ${REPEAT_TIMES}"
echo "  results_dir = ${RESULTS_DIR}"
echo "  workspace_dir = ${WORKSPACE_DIR}"
echo "  per_device_train_batch_size = ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  gradient_accumulation_steps = ${GRADIENT_ACCUMULATION_STEPS}"
echo "  skip_train  = ${SKIP_TRAINING}"

python "${SCRIPT_DIR}/pipeline_core.py" \
  --model_path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --gpu_ids "$GPU_IDS" \
  --task_names "$TASK_NAMES" \
  --eval_split "$EVAL_SPLIT" \
  --few_shot_k "$FEW_SHOT_K" \
  --train_size "$TRAIN_SIZE" \
  --epochs "$EPOCHS" \
  --repeat_times "$REPEAT_TIMES" \
  --results_dir "$RESULTS_DIR" \
  --workspace_dir "$WORKSPACE_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --skip_training "$SKIP_TRAINING"

python "${SCRIPT_DIR}/summarize_results_impl.py" \
  --results_dir "$RESULTS_DIR" \
  --mode "$DATASET"
