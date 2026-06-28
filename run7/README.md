# run7: Reward Shaping v2 Ablations

Purpose: test reward shaping v2, especially the more Sokoban-aware distance
signals.

Main idea:

- `box_target_distance_mode=reverse_push`: estimate box-to-target distance by
  reverse-push reachability instead of plain ground BFS.
- `agent_box_distance_mode=useful`: guide agents toward positions that can make
  the global box-target distance decrease.
- Optional `shaping_unreachable_distance` caps very large unreachable-distance
  penalties.

Important outcome:

- v2 was more theoretically faithful, but it often performed worse than the
  older plain BFS shaping in this project.
- A likely cause is reward scale and sparsity: unreachable reverse-push states
  can dominate the learning signal unless capped.

Representative files:

- `run7.sh`, `run7.1.sh`, `run7.2.sh`: early v2 variants.
- `run7.3` to `run7.7`: finish reward and credit-horizon style trials.
- `run7.8`, `run7.9`: unreachable-distance cap trials.
- `run7.10` to `run7.12`: CNN variants with v2-style settings.
