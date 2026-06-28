# run14: 13x13 Random-Padding From-Scratch Long Runs

Purpose: train 13x13-compatible models from scratch using random episode-level
padding, then run true v0/v1/v2 mix.

Families:

- `baseline-original`
- `v1-noshape`
- `v1-nodl-nopush`
- `gru-noaction`
- `gru-8action`
- `gru-1action`
- `cnn-nodl-nopush`

Pipeline:

- First 20M steps on v0 with 7x7 maps padded to 13x13.
- Then true v0/v1/v2 mix for four 5M rounds.

Key helper files:

- `run14_phase_padding13.sh`: one phase.
- `run14_family_padding13.sh`: full family pipeline.
- `run_general14.1_padding13_all.sh`: all families.

Important outcome: random padding was a hard generalization setting for MLP/GRU.
The model must learn translation invariance from sparse symbolic inputs, which
made learning much weaker than 7x7.
