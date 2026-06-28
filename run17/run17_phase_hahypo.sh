#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <variant> <alpha_label> <alpha> <phase_label> <run_name_prefix> <exp_name> [source_prefix]" >&2
  echo "Variants: baseline | v1-nodl-nopush | gru-hist8" >&2
  exit 2
fi

VARIANT=$1
ALPHA_LABEL=$2
HYBRID_ALPHA=$3
PHASE_LABEL=$4
RUN_NAME_PREFIX=$5
EXP_NAME=$6
SOURCE_PREFIX=${7:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run17_${VARIANT}_${ALPHA_LABEL}_${PHASE_LABEL}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

MODEL_ARGS=()
if [[ -n "$SOURCE_PREFIX" ]]; then
  MODEL_DIR=$(find "results/$SOURCE_PREFIX" -type d -path '*/models' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
  if [[ -z "${MODEL_DIR:-}" || ! -d "$MODEL_DIR" ]]; then
    echo "Could not find source model directory under results/$SOURCE_PREFIX" >&2
    exit 1
  fi
  echo "Loading model from $MODEL_DIR"
  MODEL_ARGS=(--model_dir "$MODEL_DIR")
fi

COMMON_ENV_ARGS=(
  --scenario TwoPlayer-Sokoban-v0
  --control_mode turn_based
  --observation_type vector
  --n_rollout_threads "${N_ROLLOUT_THREADS:-32}"
  --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS:-16}"
  --episode_length 150
  --num_env_steps 5000000
  --max_steps 150
  --dim_room 7
  --num_boxes 2
)

COMMON_ALGO_ARGS=(
  --lr "${LR:-1e-4}"
  --critic_lr "${CRITIC_LR:-3e-4}"
  --entropy_coef "${ENTROPY_COEF:-0.01}"
  --ppo_epoch "${PPO_EPOCH:-5}"
  --clip_param "${CLIP_PARAM:-0.1}"
  --hidden_sizes "${HIDDEN_SIZES:-[512, 512]}"
  --hybrid_alpha "$HYBRID_ALPHA"
  --hybrid_group_clip "${HYBRID_GROUP_CLIP:-5.0}"
  --hybrid_group_eps "${HYBRID_GROUP_EPS:-1e-5}"
  --log_interval 5
  --eval_interval "${EVAL_INTERVAL:-50}"
  --eval_episodes "${EVAL_EPISODES:-20}"
  --use_linear_lr_decay True
  --use_eval True
)

VARIANT_ARGS=()
case "$VARIANT" in
  baseline)
    VARIANT_ARGS=(
      --reward_scale 1.0
      --reward_finished 10
      --use_reward_shaping False
    )
    ;;
  v1-nodl-nopush)
    VARIANT_ARGS=(
      --reward_finished 20
      --use_reward_shaping True
      --box_target_distance_mode bfs
      --agent_box_distance_mode adjacent
      --distance_shaping_weight 0.20
      --pushability_shaping_weight 0.0
      --deadlock_penalty 0.0
      --deadlock_penalty_mode increase
      --agent_box_distance_shaping_weight 0.10
      --useful_push_shaping_weight 0.0
    )
    ;;
  gru-hist8)
    VARIANT_ARGS=(
      --reward_finished 20
      --action_history_len 8
      --use_recurrent_policy True
      --recurrent_n "${RECURRENT_N:-1}"
      --data_chunk_length "${DATA_CHUNK_LENGTH:-25}"
      --use_reward_shaping True
      --box_target_distance_mode bfs
      --agent_box_distance_mode adjacent
      --distance_shaping_weight 0.20
      --pushability_shaping_weight 0.0
      --deadlock_penalty 0.0
      --deadlock_penalty_mode increase
      --agent_box_distance_shaping_weight 0.10
      --useful_push_shaping_weight 0.0
    )
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo hahypo \
  --env sokoban \
  --exp_name "$EXP_NAME" \
  --run_name_prefix "$RUN_NAME_PREFIX" \
  "${MODEL_ARGS[@]}" \
  --cuda True \
  "${COMMON_ENV_ARGS[@]}" \
  "${COMMON_ALGO_ARGS[@]}" \
  "${VARIANT_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
