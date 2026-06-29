#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run18_family_useless_action.sh" mlp-nopush-dl02 p02 0.2 happo-useless-p02-v1-nopush-dl02
