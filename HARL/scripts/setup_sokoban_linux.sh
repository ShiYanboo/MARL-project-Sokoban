#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$ROOT_DIR/.." && pwd)"
GYM_SOKOBAN_DIR="$WORKSPACE_DIR/gym-sokoban"

# Override these from the shell when needed, e.g.:
#   ENV_NAME=harl-sokoban PYTHON_VERSION=3.10 TORCH_CHANNEL=cu118 bash scripts/setup_sokoban_linux.sh
#   ENV_NAME=harl-sokoban-cu126 TORCH_CHANNEL=cu126 bash scripts/setup_sokoban_linux.sh
ENV_NAME="${ENV_NAME:-harl-sokoban}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Override these from the shell when needed, e.g.:
#   TORCH_CHANNEL=cu126 bash scripts/setup_sokoban_linux.sh
#   TORCH_CHANNEL=cpu bash scripts/setup_sokoban_linux.sh
TORCH_CHANNEL="${TORCH_CHANNEL:-cu118}"

case "$TORCH_CHANNEL" in
  cu118|cu121|cu124|cu126|cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CHANNEL}"
    ;;
  *)
    echo "Unsupported TORCH_CHANNEL: $TORCH_CHANNEL"
    echo "Choose one of: cu118, cu121, cu124, cu126, cpu"
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH. Please load conda first, then rerun this script."
  exit 1
fi

if [[ -z "${TORCH_VERSION:-}" || -z "${TORCHVISION_VERSION:-}" || -z "${TORCHAUDIO_VERSION:-}" ]]; then
  case "$TORCH_CHANNEL" in
    cu118|cu121|cpu)
      TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.2}"
      TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.2}"
      ;;
    cu124)
      TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
      TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
      ;;
    cu126)
      TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
      TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0}"
      ;;
  esac
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip wheel
conda run -n "$ENV_NAME" python -m pip install "setuptools==65.5.0"
conda run -n "$ENV_NAME" python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

conda run -n "$ENV_NAME" python -m pip install \
  "numpy==1.26.4" \
  "gym==0.23.1" \
  "matplotlib<3.9" \
  "tensorboard==2.13.0" \
  "tensorboardX==2.6.5" \
  "protobuf<5" \
  "setuptools==65.5.0" \
  "six>=1.16.0" \
  setproctitle \
  absl-py \
  imageio \
  requests \
  tqdm \
  pyyaml \
  pypdf

conda run -n "$ENV_NAME" python -m pip install -e "$GYM_SOKOBAN_DIR"
conda run -n "$ENV_NAME" python -m pip install -e "$ROOT_DIR"

conda run -n "$ENV_NAME" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
PY

echo "Sokoban HARL environment is ready in conda env: $ENV_NAME"
echo "Activate it with: conda activate $ENV_NAME"
