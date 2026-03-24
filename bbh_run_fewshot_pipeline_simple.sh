#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-3B-Instruct"
DATASET="bbh"
GPU_IDS="0,1,2,3"
TASK_NAMES=""
EVAL_SPLIT="test"
FEW_SHOT_K=5
TRAIN_SIZE=""
EPOCHS=3
REPEAT_TIMES=10
RESULTS_DIR="${SCRIPT_DIR}/results"
SKIP_TRAINING=false

usage() {
  cat <<EOF
Usage:
  bash run_fewshot_pipeline_simple.sh --model_path /path/to/model --dataset bbh|arc|password [options]

Required:
  --model_path PATH
  --dataset bbh|arc|password

Common options:
  --gpu_ids 0,1,2,3
  --task_names task1,task2        BBH/password only
  --eval_split validation|test    ARC only
  --few_shot_k 5
  --train_size 5                  default = few_shot_k
  --epochs 3
  --repeat_times 10
  --results_dir PATH
  --skip_training true|false

Split semantics:
  BBH/password: current task's first train_size samples are used for TTT/SFT and few-shot support;
       the remaining samples are evaluated.
  ARC: current split's first train_size samples are used for TTT/SFT and few-shot support;
       the remaining samples in that same split are evaluated.

Examples:
  bash run_fewshot_pipeline_simple.sh \\
    --model_path /mnt/model/Qwen2.5-1.5B-Instruct \\
    --dataset bbh \\
    --task_names boolean_expressions,date_understanding \\
    --gpu_ids 0,1,2,3

  bash run_fewshot_pipeline_simple.sh \\
    --model_path /mnt/model/Qwen2.5-1.5B-Instruct \\
    --dataset arc \\
    --eval_split validation \\
    --gpu_ids 0,1,2,3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --task_names) TASK_NAMES="$2"; shift 2 ;;
    --eval_split) EVAL_SPLIT="$2"; shift 2 ;;
    --few_shot_k) FEW_SHOT_K="$2"; shift 2 ;;
    --train_size) TRAIN_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --repeat_times) REPEAT_TIMES="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --skip_training) SKIP_TRAINING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$DATASET" ]]; then
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
echo "  task_names  = ${TASK_NAMES:-<all/default>}"
echo "  eval_split  = ${EVAL_SPLIT}"
echo "  few_shot_k  = ${FEW_SHOT_K}"
echo "  train_size  = ${TRAIN_SIZE}"
echo "  epochs      = ${EPOCHS}"
echo "  repeat_times = ${REPEAT_TIMES}"
echo "  results_dir = ${RESULTS_DIR}"
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
  --skip_training "$SKIP_TRAINING"

python "${SCRIPT_DIR}/summarize_results_impl.py" \
  --results_dir "$RESULTS_DIR" \
  --mode "$DATASET"