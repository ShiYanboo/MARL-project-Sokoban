#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run17_family_hahypo.sh" v1-nodl-nopush a10 1.0 hahypo-a10-v1-nodl-nopush
