#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-harl-sokoban}"
DEMO_PY="$ROOT_DIR/finalmodel/demo_finalmodel_gui.py"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/marl_sokoban_matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" "$DEMO_PY" "$@"
fi

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    exec conda run -n "$ENV_NAME" python "$DEMO_PY" "$@"
  fi
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$DEMO_PY" "$@"
fi

exec python "$DEMO_PY" "$@"
