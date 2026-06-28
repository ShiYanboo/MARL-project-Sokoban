#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" nohist 0 mixv012-r1 \
  happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1 \
  happo_turn_based_bfs_nodl_nopush_gru_nohist_mixv012_r1 \
  happo-rnn-bfs-nodl-nopush-nohist-resume20m \
  TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2
