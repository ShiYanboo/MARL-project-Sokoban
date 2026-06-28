# run12: True-Mix Padded Smoke Tests

Purpose: verify that fixed-size padded observations can support real v0/v1/v2
mixing without forcing `dim_room` and `num_boxes`.

Key idea:

- Enable `--use_mixed_obs_padding True`
- Set a maximum canvas with `--obs_dim_room 13`
- Use `scenario_pool=TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2`
- Remove fixed room-size arguments in mix mode

Files:

- `run12.1_true_mix_padded_smoke.sh`: quick smoke check.
- `run12.2_true_mix_padded_gru_hist8_5m.sh`: 5M GRU/action-history test.
- `run_general12.1_true_mix_padded.sh`: small true-mix padded pipeline.

This directory is mostly for validating plumbing, not for final results.
