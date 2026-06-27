#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/run11.1_gru_nohist_5m.sh"
bash "$SCRIPT_DIR/run11.2_gru_nohist_resume10m.sh"
bash "$SCRIPT_DIR/run11.3_gru_nohist_resume15m.sh"
bash "$SCRIPT_DIR/run11.4_gru_nohist_resume20m.sh"
bash "$SCRIPT_DIR/run11.5_gru_nohist_mixv012_r1.sh"
bash "$SCRIPT_DIR/run11.6_gru_nohist_mixv012_r2.sh"
bash "$SCRIPT_DIR/run11.7_gru_nohist_mixv012_r3.sh"
bash "$SCRIPT_DIR/run11.8_gru_nohist_mixv012_r4.sh"
