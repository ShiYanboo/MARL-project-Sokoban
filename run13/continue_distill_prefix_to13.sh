#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source_result_prefix> <student_distill_prefix> <output_name> [episodes] [target_observation_type]" >&2
  echo "Example: $0 happo-rnn-bfs-nodl-nopush-resume20m distill13-gru-8action-premix20m distill13-gru-8action-cont-bepoch3 2000 same" >&2
  exit 2
fi

SOURCE_PREFIX=$1
STUDENT_PREFIX=$2
OUTPUT_NAME=$3
EPISODES=${4:-2000}
TARGET_OBSERVATION_TYPE=${5:-same}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODEL_DIR=$(find "results/$SOURCE_PREFIX" -type d -path '*/models' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "${MODEL_DIR:-}" || ! -d "$MODEL_DIR" ]]; then
  echo "Could not find teacher model directory under results/$SOURCE_PREFIX" >&2
  exit 1
fi

STUDENT_MODEL_DIR=$(find "results/$STUDENT_PREFIX" -type d -path '*/models' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "${STUDENT_MODEL_DIR:-}" || ! -d "$STUDENT_MODEL_DIR" ]]; then
  echo "Could not find student model directory under results/$STUDENT_PREFIX" >&2
  exit 1
fi

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run13_continue_distill_${OUTPUT_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Teacher model: $MODEL_DIR"
echo "Student init: $STUDENT_MODEL_DIR"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-32} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-32} \
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-32} \
NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-32} \
PYTHONUNBUFFERED=1 python -u HARL/scripts/distill_sokoban_padding.py \
  --model-dir "$MODEL_DIR" \
  --student-model-dir "$STUDENT_MODEL_DIR" \
  --output-dir "results/$OUTPUT_NAME" \
  --target-dim-room 13 \
  --target-observation-type "$TARGET_OBSERVATION_TYPE" \
  --padding-mode random_episode \
  --episodes "$EPISODES" \
  --batch-epochs "${BATCH_EPOCHS:-3}" \
  --lr 3e-4 \
  --critic-lr 3e-4 \
  --temperature 1.0 \
  --value-loss-coef 0.5 \
  --rollout-action-mode deterministic \
  --device cuda \
  --torch-threads "${TORCH_THREADS:-32}" \
  --log-interval 25 \
  --copy-value-normalizer 2>&1 | tee -a "$LOGFILE"
