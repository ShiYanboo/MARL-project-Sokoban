#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

bash "$SCRIPT_DIR/run16.1_v1_nodl_nopush_credit_progress005.sh"
bash "$SCRIPT_DIR/run16.2_v1_nodl_nopush_credit_progress010.sh"
bash "$SCRIPT_DIR/run16.3_v1_nodl_nopush_credit_reward002.sh"
