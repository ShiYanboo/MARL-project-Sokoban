#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general14.1] baseline original, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.1_baseline_original13_full.sh"

echo "[general14.1] v1 no-shape, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.2_v1_noshape13_full.sh"

echo "[general14.1] v1 nodl-nopush, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.3_v1_nodl_nopush13_full.sh"

echo "[general14.1] GRU no action history, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.4_gru_noaction13_full.sh"

echo "[general14.1] GRU 8-action history, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.5_gru_8action13_full.sh"

echo "[general14.1] GRU 1-action history, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.6_gru_1action13_full.sh"

echo "[general14.1] CNN nodl-nopush, 13x13 random-padding"
bash "$SCRIPT_DIR/run14.7_cnn_nodl_nopush13_full.sh"
