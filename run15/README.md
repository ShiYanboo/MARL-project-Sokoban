# run15: 13x13 Top-Left Curriculum

Purpose: make 13x13 training less sparse than random padding by using stable
top-left padding during the v0 curriculum, then true v0/v1/v2 mix.

Families:

- `baseline-original`
- `v1-noshape`
- `v1-nodl-nopush`
- `v1-deadlock005`
- `v1-deadlock020`
- `gru-noaction`
- `gru-8action`
- `gru-1action`
- `cnn-nodl-nopush`

Pipeline:

- 20M v0 pretraining with top-left 13x13 padding.
- Four 5M true-mix rounds with v0/v1/v2 and top-left padding.

Key helper files:

- `run15_phase_top_left13.sh`: one phase.
- `run15_family_top_left13_curriculum.sh`: full family pipeline.
- `run_general15.1_top_left13_curriculum_all.sh`: all main families.
- `run_general15.2_top_left13_deadlock_long.sh`: deadlock-weight long variants.
- `run15_recover_after_5m_top_left13.sh`: resume one family from an already-created 5m model.
- `run_general15.3_recover_interrupted_top_left13.sh`: recover the 2026-06-27 interrupted main families after the old `eadlock_penalty_mode` shell typo.
- `run_general15.4_recover_interrupted_top_left13_parallel.sh`: same recovery as above, but launch the seven families concurrently. Use `GPU_LIST=0,1,2` to round-robin across GPUs; otherwise all jobs default to GPU 5.

Motivation: top-left padding removes the extra burden of learning translation
invariance during early curriculum. It keeps the final network 13x13-compatible
while giving denser early learning signal.
