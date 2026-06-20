# Command Templates

## 1. Create a named conda environment

```bash
conda create -y -n harl-sokoban python=3.10
```

## 2. Upgrade packaging tools inside that environment

```bash
conda run -n harl-sokoban python -m pip install --upgrade pip wheel setuptools
```

## 3. Install PyTorch with an explicit CUDA wheel channel

`cu118` example:

```bash
conda run -n harl-sokoban python -m pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

`cu121` example:

```bash
conda run -n harl-sokoban python -m pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

`cpu` example:

```bash
conda run -n harl-sokoban python -m pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cpu
```

## 4. Install base experiment packages

```bash
conda run -n harl-sokoban python -m pip install \
  gym==0.23.1 \
  tensorboardX \
  setproctitle \
  absl-py \
  imageio \
  requests \
  tqdm \
  pyyaml \
  pypdf
```

## 5. Clone repositories into one workspace

```bash
mkdir -p ~/projects/marl-sokoban
cd ~/projects/marl-sokoban
git clone <repo-a-url>
git clone <repo-b-url>
```

If the repos already exist, prefer updating or reusing them instead of recloning.

## 6. Install local repositories in editable mode

```bash
conda run -n harl-sokoban python -m pip install -e ./gym-sokoban
conda run -n harl-sokoban python -m pip install -e ./HARL
```

## 7. Verify torch and CUDA

```bash
conda run -n harl-sokoban python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
PY
```

## 8. Run a minimal verification command

```bash
conda run -n harl-sokoban python HARL/scripts/verify_sokoban_setup.py
```

## 9. Run experiments

Interactive:

```bash
conda activate harl-sokoban
cd HARL
python examples/train.py --algo happo --env sokoban --exp_name trial_happo --scenario TwoPlayer-Sokoban-v0
```

Non-interactive:

```bash
conda run -n harl-sokoban python HARL/examples/train.py --algo hasac --env sokoban --exp_name trial_hasac --scenario TwoPlayer-Sokoban-v0
```

## 10. What to report back after setup

- Environment name
- Python version
- CUDA wheel channel
- Torch version
- Which repos were cloned or reused
- Editable installs performed
- Verification command and result
- Canonical experiment command
