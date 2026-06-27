#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[general0.1] baseline original 5m"
bash "$SCRIPT_DIR/run0.1_baseline_original_5m.sh"

echo "[general0.1] baseline original resume10m"
bash "$SCRIPT_DIR/run0.2_baseline_original_resume10m.sh"

echo "[general0.1] baseline original resume15m"
bash "$SCRIPT_DIR/run0.3_baseline_original_resume15m.sh"

echo "[general0.1] baseline original resume20m"
bash "$SCRIPT_DIR/run0.4_baseline_original_resume20m.sh"

echo "[general0.1] baseline original mix v0/v1/v2 round 1"
bash "$SCRIPT_DIR/run0.5_baseline_original_mixv012_r1.sh"

echo "[general0.1] baseline original mix v0/v1/v2 round 2"
bash "$SCRIPT_DIR/run0.6_baseline_original_mixv012_r2.sh"

echo "[general0.1] baseline original mix v0/v1/v2 round 3"
bash "$SCRIPT_DIR/run0.7_baseline_original_mixv012_r3.sh"

echo "[general0.1] baseline original mix v0/v1/v2 round 4"
bash "$SCRIPT_DIR/run0.8_baseline_original_mixv012_r4.sh"
