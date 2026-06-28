# Report Key Metrics

Metrics: train/eval success rate, train/eval box completion ratio, train/eval base reward per step, critic value loss, and eval episode reward.
The dashed vertical line marks the planned 20M pretrain/mix boundary.

## Fig. 1 Original 7x7: key train/eval metrics
### baseline original
- happo-baseline-original: `results/happo-baseline-original/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original/happo-baseline-original-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-09-36-27`
- happo-baseline-original-resume10m: `results/happo-baseline-original-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_resume10m/happo-baseline-original-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-12-43-40`
- happo-baseline-original-resume15m: `results/happo-baseline-original-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_resume15m/happo-baseline-original-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-16-19-09`
- happo-baseline-original-resume20m: `results/happo-baseline-original-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_resume20m/happo-baseline-original-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-20-38-37`
- happo-baseline-original-mixv012-r1: `results/happo-baseline-original-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_mixv012_r1/happo-baseline-original-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-02-48-28`
- happo-baseline-original-mixv012-r2: `results/happo-baseline-original-mixv012-r2/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_mixv012_r2/happo-baseline-original-mixv012-r2-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-08-18-04`
- happo-baseline-original-mixv012-r3: `results/happo-baseline-original-mixv012-r3/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_mixv012_r3/happo-baseline-original-mixv012-r3-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-13-27-33`
- happo-baseline-original-mixv012-r4: `results/happo-baseline-original-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_baseline_original_mixv012_r4/happo-baseline-original-mixv012-r4-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-19-01-28`

### v1 nodl-nopush
- happo-shaped-nodl-nopush: `results/happo-shaped-nodl-nopush/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush/happo-shaped-nodl-nopush-steps5000000-len150-env0-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-23-16-51-33`
- happo-shaped-nodl-nopush-resume5m: `results/happo-shaped-nodl-nopush-resume5m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_resume5m/happo-shaped-nodl-nopush-resume5m-steps5000000-len150-env8-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-24-08-47-11`
- happo-shaped-nodl-nopush-resume10m: `results/happo-shaped-nodl-nopush-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_resume10m/happo-shaped-nodl-nopush-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-11-40-36`
- happo-shaped-nodl-nopush-resume15m: `results/happo-shaped-nodl-nopush-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_resume15m/happo-shaped-nodl-nopush-resume15m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-15-44-18`
- happo-shaped-nodl-nopush-mixv012-r1: `results/happo-shaped-nodl-nopush-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_mixv012_r1/happo-shaped-nodl-nopush-mixv012-r1-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-19-45-07`
- happo-shaped-nodl-nopush-mixv012-r2: `results/happo-shaped-nodl-nopush-mixv012-r2/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_mixv012_r2/happo-shaped-nodl-nopush-mixv012-r2-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-02-58-33`
- happo-shaped-nodl-nopush-mixv012-r3: `results/happo-shaped-nodl-nopush-mixv012-r3/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_mixv012_r3/happo-shaped-nodl-nopush-mixv012-r3-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-08-21-31`
- happo-shaped-nodl-nopush-mixv012-r4: `results/happo-shaped-nodl-nopush-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_shaped_nodl_nopush_mixv012_r4/happo-shaped-nodl-nopush-mixv012-r4-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-31-40`

### gru0 no-action-history
- happo-rnn-bfs-nodl-nopush-nohist: `results/happo-rnn-bfs-nodl-nopush-nohist/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist/happo-rnn-bfs-nodl-nopush-nohist-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-19-00-56`
- happo-rnn-bfs-nodl-nopush-nohist-resume10m: `results/happo-rnn-bfs-nodl-nopush-nohist-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_resume10m/happo-rnn-bfs-nodl-nopush-nohist-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-12-44-02`
- happo-rnn-bfs-nodl-nopush-nohist-resume15m: `results/happo-rnn-bfs-nodl-nopush-nohist-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_resume15m/happo-rnn-bfs-nodl-nopush-nohist-resume15m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-16-38-31`
- happo-rnn-bfs-nodl-nopush-nohist-resume20m: `results/happo-rnn-bfs-nodl-nopush-nohist-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_resume20m/happo-rnn-bfs-nodl-nopush-nohist-resume20m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-21-24-18`
- happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1: `results/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_mixv012_r1/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-02-59-23`
- happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2: `results/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_nohist_mixv012_r2/happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-11-50`
- happo-rnn-bfs-nodl-nopush-nohist-mixv012-r3: missing
- happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4: missing

### gru1 one-action-history
- happo-rnn-bfs-nodl-nopush-hist1-thr32: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32/happo-rnn-bfs-nodl-nopush-hist1-thr32-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-00-02`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_resume10m/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-04-49`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_resume15m/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-23-36-59`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_resume20m/happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-04-48-11`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_mixv012_r1/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-12-44-23`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_mixv012_r2/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-18-05-36`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_mixv012_r3/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-00-59-44`
- happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4: `results/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_mixv012_r4/happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-06-48-46`

### gru8 eight-action-history
- happo-rnn-bfs-nodl-nopush: `results/happo-rnn-bfs-nodl-nopush/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn/happo-rnn-bfs-nodl-nopush-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-13-41-27`
- happo-rnn-bfs-nodl-nopush-resume10m: `results/happo-rnn-bfs-nodl-nopush-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_resume10m/happo-rnn-bfs-nodl-nopush-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-25-19-02-02`
- happo-rnn-bfs-nodl-nopush-resume15m: `results/happo-rnn-bfs-nodl-nopush-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_resume15m/happo-rnn-bfs-nodl-nopush-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-01-14-09`
- happo-rnn-bfs-nodl-nopush-resume20m: `results/happo-rnn-bfs-nodl-nopush-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_resume20m/happo-rnn-bfs-nodl-nopush-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-05-37-48`
- happo-rnn-bfs-nodl-nopush-mixv012-r1: `results/happo-rnn-bfs-nodl-nopush-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_mixv012_r1/happo-rnn-bfs-nodl-nopush-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-09-37-47`
- happo-rnn-bfs-nodl-nopush-mixv012-r2: `results/happo-rnn-bfs-nodl-nopush-mixv012-r2/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_mixv012_r2/happo-rnn-bfs-nodl-nopush-mixv012-r2-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-15-21-28`
- happo-rnn-bfs-nodl-nopush-mixv012-r3: `results/happo-rnn-bfs-nodl-nopush-mixv012-r3/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_mixv012_r3/happo-rnn-bfs-nodl-nopush-mixv012-r3-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-21-56-31`
- happo-rnn-bfs-nodl-nopush-mixv012-r4: `results/happo-rnn-bfs-nodl-nopush-mixv012-r4/sokoban/TwoPlayer-Sokoban-v0/happo/happo_turn_based_bfs_nodl_nopush_rnn_mixv012_r4/happo-rnn-bfs-nodl-nopush-mixv012-r4-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-12-16`

## Fig. 2 13x13 random padding: key train/eval metrics
### 13 random baseline original
- happo13-baseline-original: `results/happo13-baseline-original/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_baseline_original_5m/happo13-baseline-original-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-42-50`
- happo13-baseline-original-resume10m: `results/happo13-baseline-original-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_baseline_original_resume10m/happo13-baseline-original-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-24-07`
- happo13-baseline-original-resume15m: `results/happo13-baseline-original-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_baseline_original_resume15m/happo13-baseline-original-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-23-44-19`
- happo13-baseline-original-resume20m: `results/happo13-baseline-original-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_baseline_original_resume20m/happo13-baseline-original-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-04-32-13`
- happo13-baseline-original-mixv012-r1: `results/happo13-baseline-original-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_baseline_original_mixv012-r1/happo13-baseline-original-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-08-20-45`
- happo13-baseline-original-mixv012-r2: missing
- happo13-baseline-original-mixv012-r3: missing
- happo13-baseline-original-mixv012-r4: missing

### 13 random v1 noshape
- happo13-v1-noshape: `results/happo13-v1-noshape/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_noshape_5m/happo13-v1-noshape-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-43-34`
- happo13-v1-noshape-resume10m: `results/happo13-v1-noshape-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_noshape_resume10m/happo13-v1-noshape-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-24-42`
- happo13-v1-noshape-resume15m: `results/happo13-v1-noshape-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_noshape_resume15m/happo13-v1-noshape-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-23-40-59`
- happo13-v1-noshape-resume20m: `results/happo13-v1-noshape-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_noshape_resume20m/happo13-v1-noshape-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-04-33-21`
- happo13-v1-noshape-mixv012-r1: `results/happo13-v1-noshape-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_noshape_mixv012-r1/happo13-v1-noshape-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-08-18-41`
- happo13-v1-noshape-mixv012-r2: missing
- happo13-v1-noshape-mixv012-r3: missing
- happo13-v1-noshape-mixv012-r4: missing

### 13 random v1 nodl-nopush
- happo13-v1-nodl-nopush: `results/happo13-v1-nodl-nopush/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_nodl_nopush_5m/happo13-v1-nodl-nopush-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-45-24`
- happo13-v1-nodl-nopush-resume10m: `results/happo13-v1-nodl-nopush-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_nodl_nopush_resume10m/happo13-v1-nodl-nopush-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-38-40`
- happo13-v1-nodl-nopush-resume15m: `results/happo13-v1-nodl-nopush-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_nodl_nopush_resume15m/happo13-v1-nodl-nopush-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-00-16-18`
- happo13-v1-nodl-nopush-resume20m: `results/happo13-v1-nodl-nopush-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_nodl_nopush_resume20m/happo13-v1-nodl-nopush-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-04-56-07`
- happo13-v1-nodl-nopush-mixv012-r1: `results/happo13-v1-nodl-nopush-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_v1_nodl_nopush_mixv012-r1/happo13-v1-nodl-nopush-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-08-57-32`
- happo13-v1-nodl-nopush-mixv012-r2: missing
- happo13-v1-nodl-nopush-mixv012-r3: missing
- happo13-v1-nodl-nopush-mixv012-r4: missing

### 13 random gru0 no-action-history
- happo13-gru-noaction: `results/happo13-gru-noaction/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_noaction_5m/happo13-gru-noaction-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-46-02`
- happo13-gru-noaction-resume10m: `results/happo13-gru-noaction-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_noaction_resume10m/happo13-gru-noaction-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-13-19`
- happo13-gru-noaction-resume15m: `results/happo13-gru-noaction-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_noaction_resume15m/happo13-gru-noaction-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-22-13-04`
- happo13-gru-noaction-resume20m: `results/happo13-gru-noaction-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_noaction_resume20m/happo13-gru-noaction-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-02-21-39`
- happo13-gru-noaction-mixv012-r1: `results/happo13-gru-noaction-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_noaction_mixv012-r1/happo13-gru-noaction-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-51-09`
- happo13-gru-noaction-mixv012-r2: missing
- happo13-gru-noaction-mixv012-r3: missing
- happo13-gru-noaction-mixv012-r4: missing

### 13 random gru1 one-action-history
- happo13-gru-1action: `results/happo13-gru-1action/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_1action_5m/happo13-gru-1action-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-47-00`
- happo13-gru-1action-resume10m: `results/happo13-gru-1action-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_1action_resume10m/happo13-gru-1action-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-15-45`
- happo13-gru-1action-resume15m: `results/happo13-gru-1action-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_1action_resume15m/happo13-gru-1action-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-22-12-07`
- happo13-gru-1action-resume20m: `results/happo13-gru-1action-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_1action_resume20m/happo13-gru-1action-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-02-20-28`
- happo13-gru-1action-mixv012-r1: `results/happo13-gru-1action-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_1action_mixv012-r1/happo13-gru-1action-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-49-42`
- happo13-gru-1action-mixv012-r2: missing
- happo13-gru-1action-mixv012-r3: missing
- happo13-gru-1action-mixv012-r4: missing

### 13 random gru8 eight-action-history
- happo13-gru-8action: `results/happo13-gru-8action/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_8action_5m/happo13-gru-8action-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-46-39`
- happo13-gru-8action-resume10m: `results/happo13-gru-8action-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_8action_resume10m/happo13-gru-8action-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-13-13`
- happo13-gru-8action-resume15m: `results/happo13-gru-8action-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_8action_resume15m/happo13-gru-8action-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-22-11-56`
- happo13-gru-8action-resume20m: `results/happo13-gru-8action-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_8action_resume20m/happo13-gru-8action-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-02-23-35`
- happo13-gru-8action-mixv012-r1: `results/happo13-gru-8action-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_gru_8action_mixv012-r1/happo13-gru-8action-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-06-00-36`
- happo13-gru-8action-mixv012-r2: missing
- happo13-gru-8action-mixv012-r3: missing
- happo13-gru-8action-mixv012-r4: missing

### 13 random cnn nodl-nopush
- happo13-cnn-nodl-nopush: `results/happo13-cnn-nodl-nopush/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_cnn_nodl_nopush_5m/happo13-cnn-nodl-nopush-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-14-47-27`
- happo13-cnn-nodl-nopush-resume10m: `results/happo13-cnn-nodl-nopush-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_cnn_nodl_nopush_resume10m/happo13-cnn-nodl-nopush-resume10m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-26-18-50-50`
- happo13-cnn-nodl-nopush-resume15m: `results/happo13-cnn-nodl-nopush-resume15m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_cnn_nodl_nopush_resume15m/happo13-cnn-nodl-nopush-resume15m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-00-49-49`
- happo13-cnn-nodl-nopush-resume20m: `results/happo13-cnn-nodl-nopush-resume20m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_cnn_nodl_nopush_resume20m/happo13-cnn-nodl-nopush-resume20m-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-41-09`
- happo13-cnn-nodl-nopush-mixv012-r1: `results/happo13-cnn-nodl-nopush-mixv012-r1/sokoban/TwoPlayer-Sokoban-v0/happo/happo_padding13_cnn_nodl_nopush_mixv012-r1/happo13-cnn-nodl-nopush-mixv012-r1-steps5000000-len150-env4-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-09-54-26`
- happo13-cnn-nodl-nopush-mixv012-r2: missing
- happo13-cnn-nodl-nopush-mixv012-r3: missing
- happo13-cnn-nodl-nopush-mixv012-r4: missing

## Fig. 3 13x13 top-left padding: key train/eval metrics
### 13tl baseline original
- happo13tl-baseline-original: `results/happo13tl-baseline-original/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_baseline_original_5m/happo13tl-baseline-original-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-52-28`
- happo13tl-baseline-original-resume10m: `results/happo13tl-baseline-original-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_baseline_original_resume10m/happo13tl-baseline-original-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-17-30`
- happo13tl-baseline-original-resume15m: missing
- happo13tl-baseline-original-resume20m: missing
- happo13tl-baseline-original-mixv012-r1: missing
- happo13tl-baseline-original-mixv012-r2: missing
- happo13tl-baseline-original-mixv012-r3: missing
- happo13tl-baseline-original-mixv012-r4: missing

### 13tl v1 nodl-nopush
- happo13tl-v1-nodl-nopush: `results/happo13tl-v1-nodl-nopush/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_v1_nodl_nopush_5m/happo13tl-v1-nodl-nopush-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-53-14`
- happo13tl-v1-nodl-nopush-resume10m: `results/happo13tl-v1-nodl-nopush-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_v1_nodl_nopush_resume10m/happo13tl-v1-nodl-nopush-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-17-30`
- happo13tl-v1-nodl-nopush-resume15m: missing
- happo13tl-v1-nodl-nopush-resume20m: missing
- happo13tl-v1-nodl-nopush-mixv012-r1: missing
- happo13tl-v1-nodl-nopush-mixv012-r2: missing
- happo13tl-v1-nodl-nopush-mixv012-r3: missing
- happo13tl-v1-nodl-nopush-mixv012-r4: missing

### 13tl gru0 no-action-history
- happo13tl-gru-noaction: `results/happo13tl-gru-noaction/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_noaction_5m/happo13tl-gru-noaction-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-53-29`
- happo13tl-gru-noaction-resume10m: `results/happo13tl-gru-noaction-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_noaction_resume10m/happo13tl-gru-noaction-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-17-30`
- happo13tl-gru-noaction-resume15m: missing
- happo13tl-gru-noaction-resume20m: missing
- happo13tl-gru-noaction-mixv012-r1: missing
- happo13tl-gru-noaction-mixv012-r2: missing
- happo13tl-gru-noaction-mixv012-r3: missing
- happo13tl-gru-noaction-mixv012-r4: missing

### 13tl gru1 one-action-history
- happo13tl-gru-1action: `results/happo13tl-gru-1action/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_1action_5m/happo13tl-gru-1action-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-54-02`
- happo13tl-gru-1action-resume10m: `results/happo13tl-gru-1action-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_1action_resume10m/happo13tl-gru-1action-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-17-30`
- happo13tl-gru-1action-resume15m: missing
- happo13tl-gru-1action-resume20m: missing
- happo13tl-gru-1action-mixv012-r1: missing
- happo13tl-gru-1action-mixv012-r2: missing
- happo13tl-gru-1action-mixv012-r3: missing
- happo13tl-gru-1action-mixv012-r4: missing

### 13tl gru8 eight-action-history
- happo13tl-gru-8action: `results/happo13tl-gru-8action/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_8action_5m/happo13tl-gru-8action-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-27-05-53-41`
- happo13tl-gru-8action-resume10m: `results/happo13tl-gru-8action-resume10m/sokoban/TwoPlayer-Sokoban-v0/happo/happo_top_left13_gru_8action_resume10m/happo13tl-gru-8action-resume10m-steps5000000-len150-env5-lr1e-4-vlr3e-4-ppoepoch5-clip0.1-evalepi20-seed1-2026-06-28-08-17-30`
- happo13tl-gru-8action-resume15m: missing
- happo13tl-gru-8action-resume20m: missing
- happo13tl-gru-8action-mixv012-r1: missing
- happo13tl-gru-8action-mixv012-r2: missing
- happo13tl-gru-8action-mixv012-r3: missing
- happo13tl-gru-8action-mixv012-r4: missing

