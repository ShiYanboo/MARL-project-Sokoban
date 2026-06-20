$ErrorActionPreference = "Stop"
python examples/train.py `
  --algo happo `
  --env sokoban `
  --exp_name sokoban_happo `
  --scenario "'TwoPlayer-Sokoban-v0'" `
  --n_rollout_threads 2 `
  --n_eval_rollout_threads 2 `
  --episode_length 100 `
  --num_env_steps 2000 `
  --eval_interval 5 `
  --eval_episodes 2 `
  --log_interval 1 `
  --cuda False
