#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/mnt/jinbo/RLRM/model/Qwen/Qwen2.5-1.5B-Instruct"
DATASET="bbh"

# 可以在四个脚本里分别设置，例如 bbh=0,1,2,3 arc=4,5,6,7
GPU_IDS="4,5,6,7"
TRAIN_GPU_IDS=""
INFERENCE_GPU_ID=""

TASK_NAMES=""
EVAL_SPLIT="test"
FEW_SHOT_K=5
TRAIN_SIZE=""
EPOCHS=3
REPEAT_TIMES=10
RESULTS_DIR="${SCRIPT_DIR}/results"
WORKSPACE_DIR="${SCRIPT_DIR}/workspace_fewshot"
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_LORA_RANK=64
ALLOW_MERGE_FALLBACK=true
KEEP_INFER_ENGINE_ALIVE=true
KEEP_INFER_ENGINE_ACROSS_TASKS=true
SKIP_TRAINING=false

usage() {
  cat <<EOF
Usage:
  bash bbh_run_fewshot_pipeline_adapter.sh --model_path /path/to/model [options]

Common options:
  --gpu_ids 0,1,2,3
  --train_gpu_ids 1,2,3
  --inference_gpu_id 0
  --task_names task1,task2
  --few_shot_k 5
  --train_size 5
  --epochs 3
  --repeat_times 10
  --results_dir PATH
  --workspace_dir PATH
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 4
  --max_lora_rank 64
  --allow_merge_fallback true|false
  --keep_infer_engine_alive true|false
  --keep_infer_engine_across_tasks true|false
  --skip_training true|false
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --dataset) echo "This script is fixed to dataset=bbh. Please remove --dataset."; exit 1 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --train_gpu_ids) TRAIN_GPU_IDS="$2"; shift 2 ;;
    --inference_gpu_id) INFERENCE_GPU_ID="$2"; shift 2 ;;
    --task_names) TASK_NAMES="$2"; shift 2 ;;
    --eval_split) EVAL_SPLIT="$2"; shift 2 ;;
    --few_shot_k) FEW_SHOT_K="$2"; shift 2 ;;
    --train_size) TRAIN_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --repeat_times) REPEAT_TIMES="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --workspace_dir) WORKSPACE_DIR="$2"; shift 2 ;;
    --per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --max_lora_rank) MAX_LORA_RANK="$2"; shift 2 ;;
    --allow_merge_fallback) ALLOW_MERGE_FALLBACK="$2"; shift 2 ;;
    --keep_infer_engine_alive) KEEP_INFER_ENGINE_ALIVE="$2"; shift 2 ;;
    --keep_infer_engine_across_tasks) KEEP_INFER_ENGINE_ACROSS_TASKS="$2"; shift 2 ;;
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

echo "Run adapter pipeline (BBH)"
echo "  model_path  = ${MODEL_PATH}"
echo "  dataset     = ${DATASET}"
echo "  gpu_pool    = ${GPU_IDS:-<auto-detect>}"
echo "  train_gpu_ids = ${TRAIN_GPU_IDS:-<auto>}"
echo "  inference_gpu_id = ${INFERENCE_GPU_ID}"
echo "  task_names  = ${TASK_NAMES:-<all/default>}"
echo "  eval_split  = ${EVAL_SPLIT}"
echo "  few_shot_k  = ${FEW_SHOT_K}"
echo "  train_size  = ${TRAIN_SIZE}"
echo "  epochs      = ${EPOCHS}"
echo "  repeat_times = ${REPEAT_TIMES}"
echo "  results_dir = ${RESULTS_DIR}"
echo "  workspace_dir = ${WORKSPACE_DIR}"
echo "  per_device_train_batch_size = ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  gradient_accumulation_steps = ${GRADIENT_ACCUMULATION_STEPS}"
echo "  max_lora_rank = ${MAX_LORA_RANK}"
echo "  allow_merge_fallback = ${ALLOW_MERGE_FALLBACK}"
echo "  keep_infer_engine_alive = ${KEEP_INFER_ENGINE_ALIVE}"
echo "  keep_infer_engine_across_tasks = ${KEEP_INFER_ENGINE_ACROSS_TASKS}"
echo "  skip_train  = ${SKIP_TRAINING}"

python "${SCRIPT_DIR}/pipeline_core_adapter.py" \
  --model_path "$MODEL_PATH" \
  --dataset "$DATASET" \
  --gpu_ids "$GPU_IDS" \
  --train_gpu_ids "$TRAIN_GPU_IDS" \
  --inference_gpu_id "$INFERENCE_GPU_ID" \
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
  --max_lora_rank "$MAX_LORA_RANK" \
  --allow_merge_fallback "$ALLOW_MERGE_FALLBACK" \
  --keep_infer_engine_alive "$KEEP_INFER_ENGINE_ALIVE" \
  --keep_infer_engine_across_tasks "$KEEP_INFER_ENGINE_ACROSS_TASKS" \
  --skip_training "$SKIP_TRAINING"

python "${SCRIPT_DIR}/summarize_results_impl.py" \
  --results_dir "$RESULTS_DIR" \
  --mode "$DATASET"
