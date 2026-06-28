# run10: v2 Cap12 Long Continuation

Purpose: continue the v2 reward-shaping family where unreachable reverse-push
distance is capped around `12`.

Pipeline:

- `run10.1` to `run10.3`: resume the 7x7 v2-cap12 family.
- `run10.4` to `run10.7`: pseudo-mix rounds r1-r4.
- `run_general10.1.sh`: run the whole chain.

Use this as the main long-run record for the v2 cap experiment. It is still a
7x7-style pseudo-mix chain unless the script explicitly removes fixed room
settings.
