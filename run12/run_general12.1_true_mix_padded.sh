#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general12.1] true v0/v1/v2 mixed padded smoke"
bash "$SCRIPT_DIR/run12.1_true_mix_padded_smoke.sh"

echo "[general12.1] true v0/v1/v2 mixed padded GRU hist8 5m"
bash "$SCRIPT_DIR/run12.2_true_mix_padded_gru_hist8_5m.sh"
