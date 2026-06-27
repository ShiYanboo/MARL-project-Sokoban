#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/true_mix_from_prefix_to13.sh" \
  "${SOURCE_DISTILL_PREFIX:-distill13-gru-1action-premix20m-cont-bepoch3}" \
  happo-true-mix13-gru-1action-r1 \
  happo_true_mix13_gru_1action_r1
