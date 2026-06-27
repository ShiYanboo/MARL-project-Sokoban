#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/run8.25_deadlock005_5m.sh"
bash "$SCRIPT_DIR/run8.26_deadlock010_5m.sh"
bash "$SCRIPT_DIR/run8.27_deadlock020_5m.sh"
bash "$SCRIPT_DIR/run8.28_deadlock040_5m.sh"
