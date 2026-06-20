$ErrorActionPreference = "Stop"
python examples/train.py `
  --algo hasac `
  --env sokoban `
  --exp_name sokoban_hasac `
  --scenario "'TwoPlayer-Sokoban-v0'" `
  --n_rollout_threads 2 `
  --n_eval_rollout_threads 2 `
  --num_env_steps 4000 `
  --warmup_steps 200 `
  --train_interval 20 `
  --update_per_train 1 `
  --eval_interval 200 `
  --eval_episodes 2 `
  --batch_size 64 `
  --buffer_size 5000 `
  --n_step 5 `
  --cuda False
