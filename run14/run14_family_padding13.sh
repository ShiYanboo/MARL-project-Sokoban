#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <family> <base_prefix>" >&2
  exit 2
fi

FAMILY=$1
BASE_PREFIX=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[run14] $FAMILY: 13x13 random-padding v0 pretrain 5m"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" 5m "$BASE_PREFIX"

echo "[run14] $FAMILY: resume to 10m"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" resume10m "${BASE_PREFIX}-resume10m" "$BASE_PREFIX"

echo "[run14] $FAMILY: resume to 15m"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" resume15m "${BASE_PREFIX}-resume15m" "${BASE_PREFIX}-resume10m"

echo "[run14] $FAMILY: resume to 20m"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" resume20m "${BASE_PREFIX}-resume20m" "${BASE_PREFIX}-resume15m"

echo "[run14] $FAMILY: true padded mix012 round 1"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" mixv012-r1 "${BASE_PREFIX}-mixv012-r1" "${BASE_PREFIX}-resume20m" true

echo "[run14] $FAMILY: true padded mix012 round 2"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" mixv012-r2 "${BASE_PREFIX}-mixv012-r2" "${BASE_PREFIX}-mixv012-r1" true

echo "[run14] $FAMILY: true padded mix012 round 3"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" mixv012-r3 "${BASE_PREFIX}-mixv012-r3" "${BASE_PREFIX}-mixv012-r2" true

echo "[run14] $FAMILY: true padded mix012 round 4"
bash "$SCRIPT_DIR/run14_phase_padding13.sh" "$FAMILY" mixv012-r4 "${BASE_PREFIX}-mixv012-r4" "${BASE_PREFIX}-mixv012-r3" true
