# run11: GRU Action-History Follow-Up

Purpose: make the GRU/action-history comparison fair by filling missing 5M
continuation and pseudo-mix stages.

Core helper:

- `run_gru_ablation_phase.sh`: shared launcher for GRU variants. It fixes the
  old BFS shaping semantics, uses MLP observation vectors plus optional action
  history, and sets `use_recurrent_policy=True`.

Families:

- `run_general11.1_gru_nohist.sh`: original no-action-history GRU chain. This
  path uses the `*-nohist-thr32` prefix and may not match older completed
  `*-nohist` results.
- `run_general11.2_gru_hist1.sh`: one-action-history GRU chain.
- `run_general11.3_gru_action_ablation_all.sh`: action-history ablation bundle.
- `run_general11.4_gru_nohist_missing_from_existing.sh`: continue from the
  already completed `happo-rnn-bfs-nodl-nopush-nohist` 5M model and fill the
  missing 10M/15M/20M plus pseudo-mix stages.
- `run_general11.5_gru_hist1_missing_pseudomix.sh`: fill pseudo-mix for the
  already completed one-action-history 20M chain.
- `run_general11.6_gru0_gru1_missing_pseudomix.sh`: run both missing-fill chains
  sequentially.

Important caveat: the mix stages here are still the historical 7x7 pseudo-mix,
because fixed `dim_room=7` and `num_boxes=2` are kept.
