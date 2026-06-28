# run8: Old BFS Shaping Continuation and Pseudo-Mix

Purpose: continue the best early plain-BFS shaping variants and then run old
`mixv012` pseudo-mix rounds.

All run8 scripts explicitly keep the old reward semantics:

```bash
--box_target_distance_mode bfs
--agent_box_distance_mode adjacent
```

Families:

- `run_general8.1.sh`: strong potential shaping.
- `run_general8.2.sh`: no deadlock.
- `run_general8.3.sh`: no deadlock and no pushability. This became the strongest
  simple shaping family in many plots.
- `run_general8.4.sh`: no shaping except larger finish reward.
- `run_general8.5.sh`: run all four families.
- `run_general8.5_deadlock_sweep.sh`: short deadlock-weight sweep.

Important caveat: these `mixv012` runs are pseudo-mix, not true mix. They use
`scenario_pool`, but still force `dim_room=7` and `num_boxes=2`.
