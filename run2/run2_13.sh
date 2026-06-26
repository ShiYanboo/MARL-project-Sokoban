#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run2_13_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name happo_turn_based_easy \
  --run_name_prefix happo-base \
  --cuda True \
  --control_mode turn_based \
  --n_rollout_threads 16 \
  --n_eval_rollout_threads 4 \
  --episode_length 150 \
  --num_env_steps 10000000 \
  --max_steps 150 \
  --dim_room 7 \
  --num_boxes 2 \
  --lr 1e-4 \
  --critic_lr 3e-4 \
  --entropy_coef 0.01 \
  --ppo_epoch 8 \
  --clip_param 0.1 \
  --hidden_sizes "[256, 256]" \
  --log_interval 5 \
  --eval_interval 25 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
