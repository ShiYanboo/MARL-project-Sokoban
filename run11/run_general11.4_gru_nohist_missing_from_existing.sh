#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS:-32}
export N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS:-16}

bash "$SCRIPT_DIR/run11.19_gru_nohist_resume10m_from_existing.sh"
bash "$SCRIPT_DIR/run11.20_gru_nohist_resume15m_from_existing.sh"
bash "$SCRIPT_DIR/run11.21_gru_nohist_resume20m_from_existing.sh"
bash "$SCRIPT_DIR/run11.22_gru_nohist_mixv012_r1_from_existing.sh"
bash "$SCRIPT_DIR/run11.23_gru_nohist_mixv012_r2_from_existing.sh"
bash "$SCRIPT_DIR/run11.24_gru_nohist_mixv012_r3_from_existing.sh"
bash "$SCRIPT_DIR/run11.25_gru_nohist_mixv012_r4_from_existing.sh"
