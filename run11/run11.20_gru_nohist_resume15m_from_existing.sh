#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" nohist 0 resume15m \
  happo-rnn-bfs-nodl-nopush-nohist-resume15m \
  happo_turn_based_bfs_nodl_nopush_gru_nohist_resume15m \
  happo-rnn-bfs-nodl-nopush-nohist-resume10m
