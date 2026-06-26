#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$SCRIPT_DIR/run9.9_nodl_nopush_rnn_resume10m.sh"
bash "$SCRIPT_DIR/run9.10_nodl_nopush_rnn_resume15m.sh"
bash "$SCRIPT_DIR/run9.11_nodl_nopush_rnn_resume20m.sh"
bash "$SCRIPT_DIR/run9.12_nodl_nopush_rnn_mixv012_r1.sh"
bash "$SCRIPT_DIR/run9.13_nodl_nopush_rnn_mixv012_r2.sh"
bash "$SCRIPT_DIR/run9.14_nodl_nopush_rnn_mixv012_r3.sh"
bash "$SCRIPT_DIR/run9.15_nodl_nopush_rnn_mixv012_r4.sh"
