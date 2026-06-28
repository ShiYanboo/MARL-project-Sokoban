#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[run17] HAHyPO baseline alpha=0.2"
bash "$SCRIPT_DIR/run17.1_baseline_a02.sh"
echo "[run17] HAHyPO baseline alpha=0.5"
bash "$SCRIPT_DIR/run17.2_baseline_a05.sh"
echo "[run17] HAHyPO baseline alpha=1.0"
bash "$SCRIPT_DIR/run17.3_baseline_a10.sh"

echo "[run17] HAHyPO v1-nodl-nopush alpha=0.2"
bash "$SCRIPT_DIR/run17.4_v1_nodl_nopush_a02.sh"
echo "[run17] HAHyPO v1-nodl-nopush alpha=0.5"
bash "$SCRIPT_DIR/run17.5_v1_nodl_nopush_a05.sh"
echo "[run17] HAHyPO v1-nodl-nopush alpha=1.0"
bash "$SCRIPT_DIR/run17.6_v1_nodl_nopush_a10.sh"

echo "[run17] HAHyPO GRU hist8 alpha=0.2"
bash "$SCRIPT_DIR/run17.7_gru_hist8_a02.sh"
echo "[run17] HAHyPO GRU hist8 alpha=0.5"
bash "$SCRIPT_DIR/run17.8_gru_hist8_a05.sh"
echo "[run17] HAHyPO GRU hist8 alpha=1.0"
bash "$SCRIPT_DIR/run17.9_gru_hist8_a10.sh"
