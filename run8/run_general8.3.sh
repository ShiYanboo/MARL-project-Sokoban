#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general8.3] nodl-nopush resume10m/resume15m"
bash "$SCRIPT_DIR/run8.5.sh"
bash "$SCRIPT_DIR/run8.6.sh"

echo "[general8.3] nodl-nopush mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run8.17.sh"

echo "[general8.3] nodl-nopush mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run8.18.sh"

echo "[general8.3] nodl-nopush mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run8.19.sh"

echo "[general8.3] nodl-nopush mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run8.20.sh"
