#!/usr/bin/env bash
set -euo pipefail

# ensure log directory exists
LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run1_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name server_happo \
  --cuda True \
  --n_rollout_threads 16 \
  --n_eval_rollout_threads 2 \
  --episode_length 100 \
  --num_env_steps 10000000 \
  --lr 5e-4 \
  --critic_lr 5e-5 \
  --entropy_coef 0.005 \
  --ppo_epoch 25 \
  --hidden_sizes "[256, 256]" \
  --log_interval 5 \
  --eval_interval 25 \
  --eval_episodes 10 \
  --use_eval True 2>&1 | tee -a "$LOGFILE"