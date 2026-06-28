#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Parallel recovery for the 2026-06-27 run15 interruption where an old shell
# line ended as `eadlock_penalty_mode` after the 5m models had been saved.
#
# Defaults put all jobs on GPU 5, matching the run15 convention. To spread jobs:
#   GPU_LIST=0,1,2,3,4,5,6 bash run15/run_general15.4_recover_interrupted_top_left13_parallel.sh
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-5}"

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

LOG_DIR=log/run15_recover_parallel_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

pids=()
for i in "${!families[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  family=${families[$i]}
  prefix=${prefixes[$i]}
  logfile="$LOG_DIR/${family}.log"

  echo "[run15 parallel recovery] starting $family from $prefix on GPU $gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
      bash "$SCRIPT_DIR/run15_recover_after_5m_top_left13.sh" "$family" "$prefix"
  ) > "$logfile" 2>&1 &
  pids+=("$!")
done

echo "[run15 parallel recovery] logs: $LOG_DIR"

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[run15 parallel recovery] finished ${families[$i]}"
  else
    echo "[run15 parallel recovery] FAILED ${families[$i]} (see $LOG_DIR/${families[$i]}.log)" >&2
    failed=1
  fi
done

exit "$failed"
