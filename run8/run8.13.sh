#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run8.13_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

MODEL_DIR=""
for PREFIX in happo-shaped-nodeadlock-resume15m happo-shaped-nodeadlock-resume10m happo-shaped-nodeadlock-resume5m; do
  MODEL_DIR=$(find "results/$PREFIX" -path '*/models' -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
  if [[ -n "${MODEL_DIR:-}" && -d "$MODEL_DIR" ]]; then
    break
  fi
done
if [[ -z "${MODEL_DIR:-}" || ! -d "$MODEL_DIR" ]]; then
  echo "Could not find source model directory under nodeadlock resume15m/resume10m/resume5m" >&2
  exit 1
fi
echo "Loading model from $MODEL_DIR"

CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --scenario_pool TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2 \
  --exp_name happo_turn_based_shaped_nodeadlock_mixv012_r1 \
  --run_name_prefix happo-shaped-nodeadlock-mixv012-r1 \
  --model_dir "$MODEL_DIR" \
  --cuda True \
  --control_mode turn_based \
  --observation_type vector \
  --n_rollout_threads 32 \
  --n_eval_rollout_threads 16 \
  --episode_length 150 \
  --num_env_steps 5000000 \
  --max_steps 150 \
  --dim_room 7 \
  --num_boxes 2 \
  --reward_finished 20 \
  --lr 1e-4 \
  --critic_lr 3e-4 \
  --entropy_coef 0.01 \
  --ppo_epoch 5 \
  --clip_param 0.1 \
  --hidden_sizes "[512, 512]" \
  --use_reward_shaping True \
  --box_target_distance_mode bfs \
  --agent_box_distance_mode adjacent \
  --distance_shaping_weight 0.20 \
  --pushability_shaping_weight 0.05 \
  --deadlock_penalty 0.0 \
  --deadlock_penalty_mode increase \
  --agent_box_distance_shaping_weight 0.10 \
  --log_interval 5 \
  --eval_interval 50 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
