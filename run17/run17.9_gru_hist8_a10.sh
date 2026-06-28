#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run17_family_hahypo.sh" gru-hist8 a10 1.0 hahypo-a10-gru-hist8
