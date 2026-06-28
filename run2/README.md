# run2: Early HAPPO Hyperparameter Sweep

Purpose: sweep one or a small group of PPO/HAPPO hyperparameters around the early
`happo-base` setting.

Typical variables:

- actor learning rate
- critic learning rate
- PPO epoch
- clip range
- episode length
- hidden size

These scripts are mainly historical ablations. They are useful for explaining
why later experiments settled near:

- `lr=1e-4`
- `critic_lr=3e-4`
- `ppo_epoch=5`
- `clip_param=0.1`
- MLP `[512, 512]` for most shaped runs

Use them as reference, not as the current mainline.
