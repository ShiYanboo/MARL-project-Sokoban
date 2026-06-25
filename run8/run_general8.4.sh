#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general8.4] noshape resume10m/resume15m"
bash "$SCRIPT_DIR/run8.7.sh"
bash "$SCRIPT_DIR/run8.8.sh"

echo "[general8.4] noshape mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run8.21.sh"

echo "[general8.4] noshape mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run8.22.sh"

echo "[general8.4] noshape mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run8.23.sh"

echo "[general8.4] noshape mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run8.24.sh"
