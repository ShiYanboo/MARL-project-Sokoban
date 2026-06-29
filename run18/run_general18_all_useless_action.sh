#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[run18] MLP v1 nodl-nopush useless action penalty 0.1"
bash "$SCRIPT_DIR/run18.1_v1_nodl_nopush_useless01.sh"
echo "[run18] MLP v1 nodl-nopush useless action penalty 0.2"
bash "$SCRIPT_DIR/run18.2_v1_nodl_nopush_useless02.sh"
echo "[run18] GRU0 v1 nodl-nopush useless action penalty 0.1"
bash "$SCRIPT_DIR/run18.3_gru0_nodl_nopush_useless01.sh"
echo "[run18] GRU0 v1 nodl-nopush useless action penalty 0.2"
bash "$SCRIPT_DIR/run18.4_gru0_nodl_nopush_useless02.sh"
echo "[run18] MLP v1 nopush deadlock0.2 useless action penalty 0.2"
bash "$SCRIPT_DIR/run18.5_v1_nopush_dl02_useless02.sh"
