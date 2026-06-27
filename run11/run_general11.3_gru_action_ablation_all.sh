#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/run_general11.1_gru_nohist.sh"
bash "$SCRIPT_DIR/run_general11.2_gru_hist1.sh"
