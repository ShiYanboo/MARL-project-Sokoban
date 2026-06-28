#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Recovery for the 2026-06-27 run15 interruption where an old shell line
# ended as `eadlock_penalty_mode`, after the 5m model had already been saved.
families=(
  baseline-original
  v1-noshape
  v1-nodl-nopush
  gru-noaction
  gru-8action
  gru-1action
  cnn-nodl-nopush
)

prefixes=(
  happo13tl-baseline-original
  happo13tl-v1-noshape
  happo13tl-v1-nodl-nopush
  happo13tl-gru-noaction
  happo13tl-gru-8action
  happo13tl-gru-1action
  happo13tl-cnn-nodl-nopush
)

for i in "${!families[@]}"; do
  echo "[run15 recovery] starting ${families[$i]} from ${prefixes[$i]}"
  bash "$SCRIPT_DIR/run15_recover_after_5m_top_left13.sh" "${families[$i]}" "${prefixes[$i]}"
done
