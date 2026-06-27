#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general13.1] distill GRU BFS nodl nopush resume20m to 13x13 padded canvas"
bash "$SCRIPT_DIR/run13.1_distill_rnn_bfs_nodl_nopush_resume20m_to13.sh"

echo "[general13.1] distill GRU BFS nodl nopush mixv012-r4 to 13x13 padded canvas"
bash "$SCRIPT_DIR/run13.2_distill_rnn_bfs_nodl_nopush_mixv012_r4_to13.sh"
