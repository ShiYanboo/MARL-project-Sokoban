#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source_result_prefix> <run_name_prefix> <exp_name>" >&2
  exit 2
fi

SOURCE_PREFIX=$1
RUN_NAME_PREFIX=$2
EXP_NAME=$3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

MODEL_DIR=$(find "results/$SOURCE_PREFIX" -type d -path '*/models' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "${MODEL_DIR:-}" || ! -d "$MODEL_DIR" ]]; then
  echo "Could not find source model directory under results/$SOURCE_PREFIX" >&2
  exit 1
fi

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run13_true_mix_${RUN_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
echo "Loading model from $MODEL_DIR"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} \
NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1} \
PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --scenario_pool TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2 \
  --exp_name "$EXP_NAME" \
  --run_name_prefix "$RUN_NAME_PREFIX" \
  --model_dir "$MODEL_DIR" \
  --cuda True \
  --control_mode turn_based \
  --observation_type vector \
  --use_mixed_obs_padding True \
  --obs_dim_room 13 \
  --mixed_obs_padding_mode random_episode \
  --action_history_len 1 \
  --n_rollout_threads "${N_ROLLOUT_THREADS:-32}" \
  --n_eval_rollout_threads "${N_EVAL_ROLLOUT_THREADS:-16}" \
  --episode_length 150 \
  --num_env_steps 5000000 \
  --max_steps 150 \
  --reward_finished 20 \
  --lr "${LR:-1e-4}" \
  --critic_lr "${CRITIC_LR:-3e-4}" \
  --entropy_coef "${ENTROPY_COEF:-0.01}" \
  --ppo_epoch "${PPO_EPOCH:-5}" \
  --clip_param "${CLIP_PARAM:-0.1}" \
  --hidden_sizes "${HIDDEN_SIZES:-[512, 512]}" \
  --use_recurrent_policy True \
  --recurrent_n "${RECURRENT_N:-1}" \
  --data_chunk_length "${DATA_CHUNK_LENGTH:-25}" \
  --use_reward_shaping True \
  --box_target_distance_mode bfs \
  --agent_box_distance_mode adjacent \
  --distance_shaping_weight 0.20 \
  --pushability_shaping_weight 0.0 \
  --deadlock_penalty 0.0 \
  --deadlock_penalty_mode increase \
  --agent_box_distance_shaping_weight 0.10 \
  --log_interval 5 \
  --eval_interval "${EVAL_INTERVAL:-50}" \
  --eval_episodes "${EVAL_EPISODES:-20}" \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
