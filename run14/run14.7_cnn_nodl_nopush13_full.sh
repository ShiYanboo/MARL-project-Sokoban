#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/run14_family_padding13.sh" cnn-nodl-nopush happo13-cnn-nodl-nopush
