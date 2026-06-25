#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run7.5_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name happo_turn_based_shaped_v2_conservative_longcredit \
  --run_name_prefix happo-shaped-v2 \
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
  --reward_finished 100 \
  --gamma 0.995 \
  --gae_lambda 0.98 \
  --lr 5e-5 \
  --critic_lr 1.5e-4 \
  --entropy_coef 0.02 \
  --ppo_epoch 5 \
  --clip_param 0.1 \
  --hidden_sizes "[512, 512]" \
  --use_reward_shaping True \
  --distance_shaping_weight 0.08 \
  --pushability_shaping_weight 0.0 \
  --deadlock_penalty 0.0 \
  --deadlock_penalty_mode increase \
  --agent_box_distance_shaping_weight 0.02 \
  --useful_push_shaping_weight 0.0 \
  --log_interval 5 \
  --eval_interval 40 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
