#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <family> <phase> <run_name_prefix> [source_prefix] [mix]" >&2
  echo "Families: baseline-original, v1-noshape, v1-nodl-nopush, gru-noaction, gru-8action, gru-1action, cnn-nodl-nopush" >&2
  exit 2
fi

FAMILY=$1
PHASE=$2
RUN_NAME_PREFIX=$3
SOURCE_PREFIX=${4:-}
USE_MIX=${5:-false}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

OBSERVATION_TYPE=vector
ACTION_HISTORY_LEN=0
USE_RECURRENT=False
ROLLOUT_THREADS_DEFAULT=32
REWARD_FINISHED=20
USE_REWARD_SHAPING=True
BOX_TARGET_DISTANCE_MODE=bfs
AGENT_BOX_DISTANCE_MODE=adjacent
DISTANCE_WEIGHT=0.20
AGENT_BOX_WEIGHT=0.10
PUSHABILITY_WEIGHT=0.0
DEADLOCK_PENALTY=0.0
EXTRA_MODEL_ARGS=()

case "$FAMILY" in
  baseline-original)
    EXP_NAME="happo_padding13_baseline_original_${PHASE}"
    REWARD_FINISHED=10
    USE_REWARD_SHAPING=False
    ;;
  v1-noshape)
    EXP_NAME="happo_padding13_v1_noshape_${PHASE}"
    USE_REWARD_SHAPING=False
    ;;
  v1-nodl-nopush)
    EXP_NAME="happo_padding13_v1_nodl_nopush_${PHASE}"
    ;;
  gru-noaction)
    EXP_NAME="happo_padding13_gru_noaction_${PHASE}"
    USE_RECURRENT=True
    ROLLOUT_THREADS_DEFAULT=64
    ;;
  gru-8action)
    EXP_NAME="happo_padding13_gru_8action_${PHASE}"
    USE_RECURRENT=True
    ROLLOUT_THREADS_DEFAULT=64
    ACTION_HISTORY_LEN=8
    ;;
  gru-1action)
    EXP_NAME="happo_padding13_gru_1action_${PHASE}"
    USE_RECURRENT=True
    ROLLOUT_THREADS_DEFAULT=64
    ACTION_HISTORY_LEN=1
    ;;
  cnn-nodl-nopush)
    EXP_NAME="happo_padding13_cnn_nodl_nopush_${PHASE}"
    OBSERVATION_TYPE=cnn
    EXTRA_MODEL_ARGS=(
      --cnn_architecture sokoban
      --cnn_channels "[32, 64, 64]"
      --cnn_input_scale 1.0
    )
    ;;
  *)
    echo "Unknown family: $FAMILY" >&2
    exit 2
    ;;
esac

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

SCENARIO_ARGS=(--scenario TwoPlayer-Sokoban-v0)
ROOM_ARGS=(--dim_room 7 --num_boxes 2)
if [[ "$USE_MIX" == "true" ]]; then
  SCENARIO_ARGS+=(--scenario_pool TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2)
  ROOM_ARGS=()
fi

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run14_${FAMILY}_${PHASE}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} \
NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1} \
PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  "${SCENARIO_ARGS[@]}" \
  --exp_name "$EXP_NAME" \
  --run_name_prefix "$RUN_NAME_PREFIX" \
  "${MODEL_ARGS[@]}" \
  --cuda True \
  --control_mode turn_based \
  --observation_type "$OBSERVATION_TYPE" \
  --use_mixed_obs_padding True \
  --obs_dim_room 13 \
  --mixed_obs_padding_mode random_episode \
  --action_history_len "$ACTION_HISTORY_LEN" \
  --n_rollout_threads "${N_ROLLOUT_THREADS:-$ROLLOUT_THREADS_DEFAULT}" \
  --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS:-16}" \
  --episode_length 150 \
  --num_env_steps 5000000 \
  --max_steps 150 \
  "${ROOM_ARGS[@]}" \
  --reward_scale 1.0 \
  --reward_finished "$REWARD_FINISHED" \
  --lr "${LR:-1e-4}" \
  --critic_lr "${CRITIC_LR:-3e-4}" \
  --entropy_coef "${ENTROPY_COEF:-0.01}" \
  --ppo_epoch "${PPO_EPOCH:-5}" \
  --clip_param "${CLIP_PARAM:-0.1}" \
  --hidden_sizes "${HIDDEN_SIZES:-[512, 512]}" \
  --use_recurrent_policy "$USE_RECURRENT" \
  --recurrent_n "${RECURRENT_N:-1}" \
  --data_chunk_length "${DATA_CHUNK_LENGTH:-25}" \
  --use_reward_shaping "$USE_REWARD_SHAPING" \
  --box_target_distance_mode "$BOX_TARGET_DISTANCE_MODE" \
  --agent_box_distance_mode "$AGENT_BOX_DISTANCE_MODE" \
  --distance_shaping_weight "$DISTANCE_WEIGHT" \
  --pushability_shaping_weight "$PUSHABILITY_WEIGHT" \
  --deadlock_penalty "$DEADLOCK_PENALTY" \
  --deadlock_penalty_mode increase \
  --agent_box_distance_shaping_weight "$AGENT_BOX_WEIGHT" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --log_interval 5 \
  --eval_interval "${EVAL_INTERVAL:-50}" \
  --eval_episodes "${EVAL_EPISODES:-20}" \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
