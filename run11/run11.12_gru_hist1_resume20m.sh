#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" hist1 1 resume20m \
  happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m \
  happo_turn_based_bfs_nodl_nopush_gru_hist1_thr32_resume20m \
  happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m
