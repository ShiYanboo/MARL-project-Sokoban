#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general13.2] baseline original premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.3_distill_baseline_original_to13.sh"

echo "[general13.2] v1 no-shape premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.4_distill_v1_noshape_to13.sh"

echo "[general13.2] v1 no-deadlock no-push premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.5_distill_v1_nodl_nopush_to13.sh"

echo "[general13.2] GRU no action history premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.6_distill_gru_noaction_to13.sh"

echo "[general13.2] GRU 8-action history premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.7_distill_gru_8action_to13.sh"

echo "[general13.2] GRU 1-action history premix20m -> 13x13"
bash "$SCRIPT_DIR/run13.8_distill_gru_1action_to13.sh"
