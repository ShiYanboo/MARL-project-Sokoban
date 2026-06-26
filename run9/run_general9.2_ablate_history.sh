#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general9.2] bfs nodl nopush MLP + history 8"
bash "$SCRIPT_DIR/run9.5_v2_cap12_mlp_hist8.sh"

echo "[general9.2] bfs nodl nopush GRU + no action history"
bash "$SCRIPT_DIR/run9.6_v2_cap12_gru_nohist.sh"

echo "[general9.2] bfs nodl nopush GRU + history 2"
bash "$SCRIPT_DIR/run9.7_v2_cap12_gru_hist2.sh"

echo "[general9.2] bfs nodl nopush GRU + history 16"
bash "$SCRIPT_DIR/run9.8_v2_cap12_gru_hist16.sh"
