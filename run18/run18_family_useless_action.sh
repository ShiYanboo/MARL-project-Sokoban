#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <variant> <coef_label> <coef> <base_prefix>" >&2
  exit 2
fi

VARIANT=$1
COEF_LABEL=$2
COEF=$3
BASE_PREFIX=$4
SCENARIO_POOL="TwoPlayer-Sokoban-v0,TwoPlayer-Sokoban-v1,TwoPlayer-Sokoban-v2"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_SAFE=${BASE_PREFIX//-/_}

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  5m "$BASE_PREFIX" "${EXP_SAFE}_5m" none

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  resume10m "${BASE_PREFIX}-resume10m" "${EXP_SAFE}_resume10m" "$BASE_PREFIX"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  resume15m "${BASE_PREFIX}-resume15m" "${EXP_SAFE}_resume15m" "${BASE_PREFIX}-resume10m"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  resume20m "${BASE_PREFIX}-resume20m" "${EXP_SAFE}_resume20m" "${BASE_PREFIX}-resume15m"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  mixv012-r1 "${BASE_PREFIX}-mixv012-r1" "${EXP_SAFE}_mixv012_r1" "${BASE_PREFIX}-resume20m" "$SCENARIO_POOL"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  mixv012-r2 "${BASE_PREFIX}-mixv012-r2" "${EXP_SAFE}_mixv012_r2" "${BASE_PREFIX}-mixv012-r1" "$SCENARIO_POOL"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  mixv012-r3 "${BASE_PREFIX}-mixv012-r3" "${EXP_SAFE}_mixv012_r3" "${BASE_PREFIX}-mixv012-r2" "$SCENARIO_POOL"

bash "$SCRIPT_DIR/run18_phase_useless_action.sh" "$VARIANT" "$COEF_LABEL" "$COEF" \
  mixv012-r4 "${BASE_PREFIX}-mixv012-r4" "${EXP_SAFE}_mixv012_r4" "${BASE_PREFIX}-mixv012-r3" "$SCENARIO_POOL"
