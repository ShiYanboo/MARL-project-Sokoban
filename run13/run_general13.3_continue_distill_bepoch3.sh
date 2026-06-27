#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general13.3] continue baseline original distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.9_continue_baseline_original_bepoch3.sh"

echo "[general13.3] continue v1 no-shape distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.10_continue_v1_noshape_bepoch3.sh"

echo "[general13.3] continue v1 no-deadlock no-push distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.11_continue_v1_nodl_nopush_bepoch3.sh"

echo "[general13.3] continue GRU no action history distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.12_continue_gru_noaction_bepoch3.sh"

echo "[general13.3] continue GRU 8-action history distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.13_continue_gru_8action_bepoch3.sh"

echo "[general13.3] continue GRU 1-action history distill, batch-epochs=${BATCH_EPOCHS:-3}"
bash "$SCRIPT_DIR/run13.14_continue_gru_1action_bepoch3.sh"
