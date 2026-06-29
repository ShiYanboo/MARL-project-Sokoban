#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run18_family_useless_action.sh" gru0-nodl-nopush p02 0.2 happo-useless-p02-gru0-nodl-nopush
