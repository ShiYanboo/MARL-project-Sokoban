#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" hist1 1 mixv012-r3 \
  happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3 \
  happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_mixv012_r3 \
  happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2 \
  TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2
