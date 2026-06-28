#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export N_ROLLOUT_THREADS=${N_ROLLOUT_THREADS:-32}
export N_EVAL_ROLLOUT_THREADS=${N_EVAL_ROLLOUT_THREADS:-16}

bash "$SCRIPT_DIR/run_general11.4_gru_nohist_missing_from_existing.sh"
bash "$SCRIPT_DIR/run_general11.5_gru_hist1_missing_pseudomix.sh"
