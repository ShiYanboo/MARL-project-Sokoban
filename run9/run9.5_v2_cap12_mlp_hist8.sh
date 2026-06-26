#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run9.5_bfs_nodl_nopush_mlp_hist8_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --exp_name happo_turn_based_bfs_nodl_nopush_mlp_hist8 \
  --run_name_prefix happo-bfs-nodl-nopush-mlp-hist8 \
  --cuda True \
  --control_mode turn_based \
  --observation_type vector \
  --action_history_len 8 \
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
  --use_recurrent_policy False \
  --use_reward_shaping True \
  --box_target_distance_mode bfs \
  --agent_box_distance_mode adjacent \
  --distance_shaping_weight 0.20 \
  --agent_box_distance_shaping_weight 0.10 \
  --pushability_shaping_weight 0.0 \
  --deadlock_penalty 0.0 \
  --deadlock_penalty_mode increase \
  --log_interval 5 \
  --eval_interval 50 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
