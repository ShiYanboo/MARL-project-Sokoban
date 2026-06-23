#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run5_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name happo_turn_based_cnn_shaped \
  --run_name_prefix happo-cnn-shaped \
  --cuda True \
  --control_mode turn_based \
  --observation_type cnn \
  --n_rollout_threads 16 \
  --n_eval_rollout_threads 8 \
  --episode_length 150 \
  --num_env_steps 3000000 \
  --max_steps 150 \
  --dim_room 7 \
  --num_boxes 2 \
  --reward_finished 20 \
  --lr 1e-4 \
  --critic_lr 3e-4 \
  --entropy_coef 0.01 \
  --ppo_epoch 5 \
  --clip_param 0.1 \
  --hidden_sizes "[512, 256]" \
  --cnn_architecture sokoban \
  --cnn_channels "[32, 64, 64]" \
  --cnn_input_scale 1.0 \
  --use_reward_shaping True \
  --distance_shaping_weight 0.05 \
  --pushability_shaping_weight 0.02 \
  --deadlock_penalty 0.75 \
  --agent_box_distance_shaping_weight 0.005 \
  --log_interval 5 \
  --eval_interval 25 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
