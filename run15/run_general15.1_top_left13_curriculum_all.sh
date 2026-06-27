#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general15.1] baseline original, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.1_baseline_original_top_left13_curriculum.sh"

echo "[general15.1] v1 no-shape, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.2_v1_noshape_top_left13_curriculum.sh"

echo "[general15.1] v1 nodl-nopush, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.3_v1_nodl_nopush_top_left13_curriculum.sh"

echo "[general15.1] GRU no action history, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.4_gru_noaction_top_left13_curriculum.sh"

echo "[general15.1] GRU 8-action history, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.5_gru_8action_top_left13_curriculum.sh"

echo "[general15.1] GRU 1-action history, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.6_gru_1action_top_left13_curriculum.sh"

echo "[general15.1] CNN nodl-nopush, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.7_cnn_nodl_nopush_top_left13_curriculum.sh"
