#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/continue_distill_prefix_to13.sh" \
  happo-shaped-noshape-resume15m \
  distill13-v1-noshape-premix20m \
  distill13-v1-noshape-premix20m-cont-bepoch3 \
  "${EPISODES:-2000}" \
  "${TARGET_OBSERVATION_TYPE:-same}"
