#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <variant> <alpha_label> <alpha> <base_prefix>" >&2
  exit 2
fi

VARIANT=$1
ALPHA_LABEL=$2
HYBRID_ALPHA=$3
BASE_PREFIX=$4

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

EXP_SAFE=${BASE_PREFIX//-/_}

bash "$SCRIPT_DIR/run17_phase_hahypo.sh" "$VARIANT" "$ALPHA_LABEL" "$HYBRID_ALPHA" \
  5m "$BASE_PREFIX" "${EXP_SAFE}_5m"

bash "$SCRIPT_DIR/run17_phase_hahypo.sh" "$VARIANT" "$ALPHA_LABEL" "$HYBRID_ALPHA" \
  resume10m "${BASE_PREFIX}-resume10m" "${EXP_SAFE}_resume10m" "$BASE_PREFIX"

bash "$SCRIPT_DIR/run17_phase_hahypo.sh" "$VARIANT" "$ALPHA_LABEL" "$HYBRID_ALPHA" \
  resume15m "${BASE_PREFIX}-resume15m" "${EXP_SAFE}_resume15m" "${BASE_PREFIX}-resume10m"

bash "$SCRIPT_DIR/run17_phase_hahypo.sh" "$VARIANT" "$ALPHA_LABEL" "$HYBRID_ALPHA" \
  resume20m "${BASE_PREFIX}-resume20m" "${EXP_SAFE}_resume20m" "${BASE_PREFIX}-resume15m"
