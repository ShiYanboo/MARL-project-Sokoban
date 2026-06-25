#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general8.5] strongpot full pipeline"
bash "$SCRIPT_DIR/run_general8.1.sh"

echo "[general8.5] nodeadlock full pipeline"
bash "$SCRIPT_DIR/run_general8.2.sh"

echo "[general8.5] nodl-nopush full pipeline"
bash "$SCRIPT_DIR/run_general8.3.sh"

echo "[general8.5] noshape baseline full pipeline"
bash "$SCRIPT_DIR/run_general8.4.sh"
