#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general9.1] BFS strongpot RNN"
bash "$SCRIPT_DIR/run9.2_bfs_strongpot_rnn.sh"

echo "[general9.1] BFS nodl-nopush RNN"
bash "$SCRIPT_DIR/run9.3_bfs_nodl_nopush_rnn.sh"

echo "[general9.1] v2 cap12 RNN"
bash "$SCRIPT_DIR/run9.4_v2_cap12_rnn.sh"
