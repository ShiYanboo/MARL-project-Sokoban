# run0: Original Reward Baseline

Purpose: run a clean HAPPO baseline with only the original Sokoban reward.

Reward setting:

- Step penalty: `-0.1`
- Box newly on target: `+1`
- Box leaves target: `-1`
- Finish reward: original `+10`
- No reward shaping

Pipeline:

- `run0.1_baseline_original_5m.sh`: first 5M steps.
- `run0.2` to `run0.4`: resume to 10M, 15M, and 20M.
- `run0.5` to `run0.8`: pseudo-mix rounds r1-r4.
- `run_general0.1_baseline_original.sh`: run the whole chain.

Important caveat: the `mixv012` stage in this old series is a pseudo-mix because
the scripts still force `dim_room=7` and `num_boxes=2`. It does not test true
multi-size generalization.
