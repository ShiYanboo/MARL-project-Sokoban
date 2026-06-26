#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

echo "[general10.1] cap12 resume10m"
bash "$SCRIPT_DIR/run10.1_cap12_resume10m.sh"

echo "[general10.1] cap12 resume15m"
bash "$SCRIPT_DIR/run10.2_cap12_resume15m.sh"

echo "[general10.1] cap12 resume20m"
bash "$SCRIPT_DIR/run10.3_cap12_resume20m.sh"

echo "[general10.1] cap12 mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run10.4_cap12_mixv012_r1.sh"

echo "[general10.1] cap12 mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run10.5_cap12_mixv012_r2.sh"

echo "[general10.1] cap12 mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run10.6_cap12_mixv012_r3.sh"

echo "[general10.1] cap12 mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run10.7_cap12_mixv012_r4.sh"
