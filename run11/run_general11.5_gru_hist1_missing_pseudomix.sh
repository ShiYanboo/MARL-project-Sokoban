#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS:-32}
export N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS:-16}

bash "$SCRIPT_DIR/run11.13_gru_hist1_mixv012_r1.sh"
bash "$SCRIPT_DIR/run11.14_gru_hist1_mixv012_r2.sh"
bash "$SCRIPT_DIR/run11.15_gru_hist1_mixv012_r3.sh"
bash "$SCRIPT_DIR/run11.16_gru_hist1_mixv012_r4.sh"
