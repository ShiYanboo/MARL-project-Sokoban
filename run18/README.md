# run18: Useless Action Penalty

This series tests a small reward-shaping penalty for actions that are legal in
the action space but semantically useless in Sokoban.

The new environment argument is:

```bash
--useless_action_penalty <coef>
```

It defaults to `0.0`, so older scripts and currently running experiments keep
their original behavior. The penalty is only active when `--use_reward_shaping
True`.

An action is counted as useless if:

- the inactive agent tries a non-noop action;
- the active agent noops while not adjacent to any box;
- a push action has no adjacent box and degrades to a blocked move;
- the active agent tries to push a box into a blocked/out-of-bounds cell;
- the active agent tries to move into a wall or box.

The penalty is logged as:

- `sokoban/train_mean_useless_action_penalty`
- `sokoban/eval_mean_useless_action_penalty`
- `sokoban/train_useless_action_rate`
- `sokoban/eval_useless_action_rate`

## Long Runs

Each wrapper runs:

```text
5M pretrain x 4 rounds + pseudo mix v0/v1/v2 x 4 rounds
```

Default resource settings are GPU 5 and 32 rollout threads.

```bash
bash run18/run18.1_v1_nodl_nopush_useless01.sh
bash run18/run18.2_v1_nodl_nopush_useless02.sh
bash run18/run18.3_gru0_nodl_nopush_useless01.sh
bash run18/run18.4_gru0_nodl_nopush_useless02.sh
bash run18/run18.5_v1_nopush_dl02_useless02.sh
```

Override resources if needed:

```bash
CUDA_VISIBLE_DEVICES=4 N_ROLLOUT_THREADS=48 bash run18/run18.1_v1_nodl_nopush_useless01.sh
```
