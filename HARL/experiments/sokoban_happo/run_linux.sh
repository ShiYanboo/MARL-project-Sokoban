#!/usr/bin/env bash
set -euo pipefail

python examples/train.py \
  --algo happo \
  --env sokoban \
  --exp_name sokoban_happo \
  --scenario "'TwoPlayer-Sokoban-v0'" \
  --n_rollout_threads 4 \
  --n_eval_rollout_threads 4 \
  --episode_length 100 \
  --num_env_steps 20000 \
  --eval_interval 10 \
  --eval_episodes 4
