#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/run11.9_gru_hist1_5m.sh"
bash "$SCRIPT_DIR/run11.10_gru_hist1_resume10m.sh"
bash "$SCRIPT_DIR/run11.11_gru_hist1_resume15m.sh"
bash "$SCRIPT_DIR/run11.12_gru_hist1_resume20m.sh"
bash "$SCRIPT_DIR/run11.13_gru_hist1_mixv012_r1.sh"
bash "$SCRIPT_DIR/run11.14_gru_hist1_mixv012_r2.sh"
bash "$SCRIPT_DIR/run11.15_gru_hist1_mixv012_r3.sh"
bash "$SCRIPT_DIR/run11.16_gru_hist1_mixv012_r4.sh"
