#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

echo "[general13.4] 7x7 GRU + 1-action history pretrain: 5m"
bash "$REPO_ROOT/run11/run11.9_gru_hist1_5m.sh"

echo "[general13.4] 7x7 GRU + 1-action history pretrain: resume to 10m"
bash "$REPO_ROOT/run11/run11.10_gru_hist1_resume10m.sh"

echo "[general13.4] 7x7 GRU + 1-action history pretrain: resume to 15m"
bash "$REPO_ROOT/run11/run11.11_gru_hist1_resume15m.sh"

echo "[general13.4] 7x7 GRU + 1-action history pretrain: resume to 20m"
bash "$REPO_ROOT/run11/run11.12_gru_hist1_resume20m.sh"

echo "[general13.4] distill 7x7 GRU + 1-action history to 13x13"
bash "$SCRIPT_DIR/run13.8_distill_gru_1action_to13.sh"

echo "[general13.4] continue distillation with batch_epochs=3"
BATCH_EPOCHS="${BATCH_EPOCHS:-3}" bash "$SCRIPT_DIR/run13.14_continue_gru_1action_bepoch3.sh"

echo "[general13.4] true padded v0/v1/v2 mix from distilled 13x13 model: round 1"
bash "$SCRIPT_DIR/run13.15_gru_1action_distilled_true_mix_r1.sh"

echo "[general13.4] true padded v0/v1/v2 mix from distilled 13x13 model: round 2"
bash "$SCRIPT_DIR/run13.16_gru_1action_distilled_true_mix_r2.sh"

echo "[general13.4] true padded v0/v1/v2 mix from distilled 13x13 model: round 3"
bash "$SCRIPT_DIR/run13.17_gru_1action_distilled_true_mix_r3.sh"

echo "[general13.4] true padded v0/v1/v2 mix from distilled 13x13 model: round 4"
bash "$SCRIPT_DIR/run13.18_gru_1action_distilled_true_mix_r4.sh"
