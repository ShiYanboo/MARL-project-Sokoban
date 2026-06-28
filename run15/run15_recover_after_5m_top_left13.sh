#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <family> <base_prefix>" >&2
  echo "Starts from the already-created 5m model and runs resume10m/15m/20m plus mixv012 r1-r4." >&2
  exit 2
fi

FAMILY=$1
BASE_PREFIX=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! find "results/$BASE_PREFIX" -type d -path '*/models' -print -quit 2>/dev/null | grep -q .; then
  echo "Could not find a 5m source model under results/$BASE_PREFIX" >&2
  exit 1
fi

run_phase() {
  local phase=$1
  local target_prefix=$2
  local source_prefix=$3
  local use_mix=${4:-false}

  echo "[run15 recovery] $FAMILY: $phase"
  bash "$SCRIPT_DIR/run15_phase_top_left13.sh" \
    "$FAMILY" \
    "$phase" \
    "$target_prefix" \
    "$source_prefix" \
    "$use_mix"
}

echo "[run15 recovery] $FAMILY: using existing 5m prefix $BASE_PREFIX"
run_phase resume10m "${BASE_PREFIX}-resume10m" "$BASE_PREFIX"
run_phase resume15m "${BASE_PREFIX}-resume15m" "${BASE_PREFIX}-resume10m"
run_phase resume20m "${BASE_PREFIX}-resume20m" "${BASE_PREFIX}-resume15m"
run_phase mixv012-r1 "${BASE_PREFIX}-mixv012-r1" "${BASE_PREFIX}-resume20m" true
run_phase mixv012-r2 "${BASE_PREFIX}-mixv012-r2" "${BASE_PREFIX}-mixv012-r1" true
run_phase mixv012-r3 "${BASE_PREFIX}-mixv012-r3" "${BASE_PREFIX}-mixv012-r2" true
run_phase mixv012-r4 "${BASE_PREFIX}-mixv012-r4" "${BASE_PREFIX}-mixv012-r3" true
