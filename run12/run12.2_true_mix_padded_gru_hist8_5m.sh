#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

LOG_DIR=log
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run12.2_true_mix_padded_gru_hist8_5m_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOGFILE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
PYTHONUNBUFFERED=1 python -u HARL/examples/train.py \
  --algo happo \
  --env sokoban \
  --scenario TwoPlayer-Sokoban-v0 \
  --scenario_pool TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2 \
  --exp_name happo_turn_based_true_mix_padded_gru_hist8 \
  --run_name_prefix happo-true-mix-padded-gru-hist8 \
  --cuda True \
  --control_mode turn_based \
  --observation_type vector \
  --use_mixed_obs_padding True \
  --obs_dim_room 10 \
  --action_history_len 8 \
  --n_rollout_threads 32 \
  --n_eval_rollout_threads 16 \
  --episode_length 150 \
  --num_env_steps 5000000 \
  --max_steps 150 \
  --reward_finished 20 \
  --lr 1e-4 \
  --critic_lr 3e-4 \
  --entropy_coef 0.01 \
  --ppo_epoch 5 \
  --clip_param 0.1 \
  --hidden_sizes "[512, 512]" \
  --use_recurrent_policy True \
  --recurrent_n 1 \
  --data_chunk_length 25 \
  --use_reward_shaping True \
  --box_target_distance_mode bfs \
  --agent_box_distance_mode adjacent \
  --distance_shaping_weight 0.20 \
  --pushability_shaping_weight 0.0 \
  --deadlock_penalty 0.0 \
  --deadlock_penalty_mode increase \
  --agent_box_distance_shaping_weight 0.10 \
  --log_interval 5 \
  --eval_interval 50 \
  --eval_episodes 20 \
  --use_linear_lr_decay True \
  --use_eval True 2>&1 | tee -a "$LOGFILE"
