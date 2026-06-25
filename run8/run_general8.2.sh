#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general8.2] nodeadlock resume10m/resume15m"
bash "$SCRIPT_DIR/run8.3.sh"
bash "$SCRIPT_DIR/run8.4.sh"

echo "[general8.2] nodeadlock mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run8.13.sh"

echo "[general8.2] nodeadlock mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run8.14.sh"

echo "[general8.2] nodeadlock mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run8.15.sh"

echo "[general8.2] nodeadlock mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run8.16.sh"
