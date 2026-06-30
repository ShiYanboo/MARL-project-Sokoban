# Final Model

This directory contains the final submitted GRU0 checkpoint copied out of the
long 7x7 HAPPO run.

## Source

Source run:

`results/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_mixv012_r4/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-23-55-40`

Copied files:

- `config.json`
- `models/actor_agent0.pt`
- `models/actor_agent1.pt`
- `models/critic_agent.pt`
- `models/value_normalizer.pt`

## Model

- Name: `GRU0`
- Architecture: GRU encoder, no explicit action-history input
- Environment family: 7x7 `TwoPlayer-Sokoban`
- Main evaluation script: `eval_finalmodel.py`

## Evaluate On v0

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 100 \
  --device cuda \
  --seed 50000
```

## Evaluate On Config Scenario Pool

The final training run used the config scenario pool
`TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2`. To evaluate
the same scenario family and report both overall and per-scenario metrics:

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 90 \
  --device cuda \
  --seed 50000 \
  --scenario_pool config
```

For a quick smoke test:

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 5 \
  --device cpu
```

## Current v0-Only Re-evaluation

A 20-episode deterministic v0-only re-evaluation has been run with the HARL
eval-style starting seed `50000`:

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 20 \
  --device cpu \
  --seed 50000 \
  --output-json finalmodel/eval_v0_20ep_seed50000_metrics.json
```

Result:

- success rate: `0.4500`
- box completion ratio: `0.6500`
- final boxes on target: `1.3000`
- episode reward: `2.0850`
- episode base reward: `1.5350`

This is a v0-only independent evaluation. It is not the same as the original
TensorBoard mixed-evaluation scalar from the training run.

## Current Mixed Re-evaluation

A 30-episode deterministic mixed re-evaluation has also been run with
`--scenario_pool config`, giving 10 episodes for each of v0/v1/v2:

```bash
/home/ybshi/miniconda3/envs/harl-sokoban/bin/python finalmodel/eval_finalmodel.py \
  --episodes 30 \
  --device cpu \
  --seed 50000 \
  --scenario_pool config \
  --output-json finalmodel/eval_mixed_30ep_seed50000_metrics.json
```

Overall result:

- success rate: `0.6000`
- box completion ratio: `0.7500`
- final boxes on target: `1.5000`
- episode reward: `7.3967`
- episode base reward: `6.7067`

Per-scenario result:

- `TwoPlayer-Sokoban-v0`: success `0.5000`, box completion `0.6500`
- `TwoPlayer-Sokoban-v1`: success `0.7000`, box completion `0.8500`
- `TwoPlayer-Sokoban-v2`: success `0.6000`, box completion `0.7500`
