# run16: Actor-Side Credit Assignment

Purpose: test a small algorithm-side credit assignment idea without changing the
environment objective or breaking previous experiments.

Why this exists:

- In turn-based Sokoban, only one active agent can change the environment at a
  step.
- The central critic should still learn the team return, because the task is
  cooperative.
- But the active actor's policy-gradient signal can be high variance: a sparse
  team return says little about whether the current push/move was useful.
- run16 adds a tiny active-agent auxiliary credit to the actor advantage. This is
  not a new environment reward, and it is off unless `actor_credit_coef > 0`.

New knobs:

```bash
--actor_credit_mode none|active_progress|active_reward
--actor_credit_coef 0.0
```

Modes:

- `active_progress`: give the active agent a small bonus for immediate useful
  progress signals, such as positive base reward, useful-push distance delta,
  positive agent-box shaping, and moving a box.
- `active_reward`: give the active agent a small scaled copy of the team reward.
- `none`: default; exactly preserves old behavior.

The central critic always receives the original team reward via `team_reward`.
Only the active actor's advantage is adjusted.

Experiments:

- `run16.1`: `active_progress`, coefficient `0.05`.
- `run16.2`: `active_progress`, coefficient `0.10`.
- `run16.3`: `active_reward`, coefficient `0.02`.
- `run_general16_credit_assignment.sh`: run all three.

Recommended interpretation: if these three 5M runs do not clearly beat
`v1-nodl-nopush`, stop doing algorithm-side tweaks and move to report writing.
