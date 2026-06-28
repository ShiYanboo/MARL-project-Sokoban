# run13: 7x7 to 13x13 Distillation Attempts

Purpose: try to preserve long-trained 7x7 policies by distilling them into
13x13 padded-observation models before true mix training.

Main scripts:

- `distill_prefix_to13.sh`: distill a source prefix to a 13x13 target.
- `continue_distill_prefix_to13.sh`: continue a previous distillation run.
- `true_mix_from_prefix_to13.sh`: start true mix from a distilled prefix.
- `run_general13.1_distill_to13.sh`: early broad distillation batch.
- `run_general13.2_distill_selected_to13.sh`: selected-source distillation.
- `run_general13.3_continue_distill_bepoch3.sh`: longer distillation per batch.
- `run_general13.4_gru_1action_pretrain_distill_true_mix.sh`: one-action-history
  GRU pretrain, distill, then true mix.

Important outcome: distillation did not recover policy quality well enough. KL
and MSE stayed high, and evaluation metrics were poor, so later work moved to
training 13x13-compatible models from scratch.
