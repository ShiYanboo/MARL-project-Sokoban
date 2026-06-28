# run9: GRU and Action-History Ablations

Purpose: test whether recurrence or explicit action history explains the strong
GRU results.

Key switches:

- `--use_recurrent_policy True`: enables HARL's GRU recurrent actor/critic.
- `--action_history_len K`: appends the last `K` executed joint actions to the
  observation.

Main variants:

- `run9.1_smoke.sh`: quick GRU smoke test.
- `run9.2`: old BFS strongpot + GRU.
- `run9.3`: old BFS no-deadlock/no-pushability + GRU.
- `run9.4`: v2 cap12 + GRU.
- `run9.5`: MLP with action history, used to check whether history alone helps.
- `run9.6`, `run9.7`, `run9.8`: GRU with 0, 2, and 16 action-history steps.
- `run9.9` to `run9.15`: continue the selected GRU no-deadlock/no-pushability
  family and run pseudo-mix.

Interpretation: GRU plus action history looked promising in 7x7, but later
13x13/padding experiments showed that size generalization remained difficult.
