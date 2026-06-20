#!/usr/bin/env bash
set -euo pipefail

python examples/train.py \
  --algo hasac \
  --env sokoban \
  --exp_name sokoban_hasac \
  --scenario "'TwoPlayer-Sokoban-v0'" \
  --n_rollout_threads 4 \
  --n_eval_rollout_threads 4 \
  --num_env_steps 20000 \
  --warmup_steps 1000 \
  --train_interval 50 \
  --update_per_train 1 \
  --eval_interval 2000 \
  --eval_episodes 4 \
  --batch_size 256 \
  --buffer_size 50000 \
  --n_step 5
