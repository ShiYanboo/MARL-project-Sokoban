#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general15.2] v1 deadlock 0.05, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.8_v1_deadlock005_top_left13_curriculum.sh"

echo "[general15.2] v1 deadlock 0.20, top-left 13x13 curriculum"
bash "$SCRIPT_DIR/run15.9_v1_deadlock020_top_left13_curriculum.sh"
