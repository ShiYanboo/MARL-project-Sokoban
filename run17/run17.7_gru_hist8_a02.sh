#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run17_family_hahypo.sh" gru-hist8 a02 0.2 hahypo-a02-gru-hist8
