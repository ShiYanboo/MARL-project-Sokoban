#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" nohist 0 resume10m \
  happo-rnn-bfs-nodl-nopush-nohist-thr32-resume10m \
  happo_turn_based_bfs_nodl_nopush_gru_nohist_thr32_resume10m \
  happo-rnn-bfs-nodl-nopush-nohist-thr32
