# run17: HAHyPO Experiments

`run17` tests the new `hahypo` algorithm on the original 7x7 v0 Sokoban setting.

HAHyPO keeps HAPPO's clipped policy update and value critic, but changes the actor
advantage to a hybrid:

```text
A_actor = (1 - alpha) * zscore(GAE advantage)
        + alpha * zscore(trajectory return within the rollout batch)
```

This is meant to borrow the group-relative idea from GRPO without deleting the
critic. The critic still learns the normal team return, so the experiment is a
conservative credit/optimization ablation rather than a pure GRPO replacement.

## Variants

- `baseline`: original reward only, `reward_finished=10`, no shaping.
- `v1-nodl-nopush`: BFS box-target potential + adjacent agent-box potential,
  no pushability penalty and no deadlock penalty.
- `gru-hist8`: same v1-nodl-nopush reward, plus GRU policy and 8-step joint
  action history.

Each variant runs alpha in `{0.2, 0.5, 1.0}` and each script chains four v0
phases of 5e6 steps: 5m, resume10m, resume15m, resume20m.

## Commands

```bash
bash run17/run17.1_baseline_a02.sh
bash run17/run17.2_baseline_a05.sh
bash run17/run17.3_baseline_a10.sh
bash run17/run17.4_v1_nodl_nopush_a02.sh
bash run17/run17.5_v1_nodl_nopush_a05.sh
bash run17/run17.6_v1_nodl_nopush_a10.sh
bash run17/run17.7_gru_hist8_a02.sh
bash run17/run17.8_gru_hist8_a05.sh
bash run17/run17.9_gru_hist8_a10.sh
```

By default scripts use GPU 5 and 32 rollout threads. Override at launch, for
example:

```bash
CUDA_VISIBLE_DEVICES=4 N_ROLLOUT_THREADS=48 bash run17/run17.7_gru_hist8_a02.sh
```
