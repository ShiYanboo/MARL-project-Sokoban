#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/distill_prefix_to13.sh" \
  happo-rnn-bfs-nodl-nopush-resume20m \
  distill13-gru-8action-premix20m \
  "${EPISODES:-2000}" \
  "${TARGET_OBSERVATION_TYPE:-same}"
