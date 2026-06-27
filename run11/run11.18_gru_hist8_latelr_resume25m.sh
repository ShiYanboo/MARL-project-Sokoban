#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

LR=5e-5 \
CRITIC_LR=1e-4 \
PPO_EPOCH=3 \
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" hist8-latelr 8 resume25m \
  happo-rnn-bfs-nodl-nopush-latelr-resume25m \
  happo_turn_based_bfs_nodl_nopush_rnn_latelr_resume25m \
  happo-rnn-bfs-nodl-nopush-resume20m
