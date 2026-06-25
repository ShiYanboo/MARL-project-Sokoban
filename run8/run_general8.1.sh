#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general8.1] strongpot resume10m"
bash "$SCRIPT_DIR/run8.1.sh"

echo "[general8.1] strongpot resume15m"
bash "$SCRIPT_DIR/run8.2.sh"

echo "[general8.1] strongpot mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run8.9.sh"

echo "[general8.1] strongpot mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run8.10.sh"

echo "[general8.1] strongpot mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run8.11.sh"

echo "[general8.1] strongpot mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run8.12.sh"
