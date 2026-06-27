#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

HIDDEN_SIZES="[768, 768]" \
RECURRENT_N=2 \
DATA_CHUNK_LENGTH=50 \
bash "$SCRIPT_DIR/run_gru_ablation_phase.sh" hist8-bigcap 8 5m \
  happo-rnn-bfs-nodl-nopush-hist8-bigcap-thr32 \
  happo_turn_based_bfs_nodl_nopush_gru_hist8_bigcap_thr32
