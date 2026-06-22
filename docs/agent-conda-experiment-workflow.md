# Agent Conda Experiment Workflow

This document defines the default workflow that agents should follow when the task is a standard research or engineering setup: create an environment, install baseline packages, clone repos if needed, verify the toolchain, and then run experiments.

## Default Principles

- Never install experiment dependencies into `base`.
- Always use a named conda environment.
- Prefer Python `3.10` unless the repo explicitly requires another version.
- Install PyTorch first when CUDA compatibility matters.
- Use an explicit PyTorch CUDA wheel channel such as `cu118`, `cu121`, `cu124`, `cu126`, or `cpu`.
- Prefer `pip install -e <repo>` for local repositories under active development.
- Verify imports and CUDA visibility before starting training.
- Save or report the exact launch command used for experiments.

## Standard Flow

1. Inspect the workspace.
   Check whether repositories already exist, whether a conda environment already exists, and whether the task targets Linux, Windows, or both.

2. Choose an environment name.
   Prefer names such as `project-slug`, `project-slug-cu118`, or `paper-repro-2026`.

3. Create the conda environment.
   Example:

   ```bash
   conda create -y -n harl-sokoban python=3.10
   ```

4. Upgrade packaging tools inside the target environment.

   ```bash
   conda run -n harl-sokoban python -m pip install --upgrade pip wheel
   conda run -n harl-sokoban python -m pip install \
     "setuptools==65.5.0" \
     "numpy==1.26.4"
   ```

5. Install PyTorch with an explicit CUDA channel.
   Example for `cu118`:

   ```bash
   conda run -n harl-sokoban python -m pip install \
     torch==2.2.2 \
     torchvision==0.17.2 \
     torchaudio==2.2.2 \
     --index-url https://download.pytorch.org/whl/cu118
   ```

6. Install base experiment packages.

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

7. Clone repositories if they are missing.
   Keep them in one intended workspace root. Do not spread project clones across unrelated directories.

8. Install local repos in editable mode.

   ```bash
   conda run -n harl-sokoban python -m pip install -e ./gym-sokoban
   conda run -n harl-sokoban python -m pip install -e ./HARL
   ```

9. Verify setup.

   ```bash
   conda run -n harl-sokoban python - <<'PY'
   import torch
   print("torch:", torch.__version__)
   print("cuda available:", torch.cuda.is_available())
   print("torch cuda version:", torch.version.cuda)
   PY
   ```

10. Run a minimal project verification before formal experiments.
    For this repository:

   ```bash
   conda run -n harl-sokoban python HARL/scripts/verify_sokoban_setup.py
   ```

11. Start experiments only after verification passes.
    Example:

   ```bash
   conda activate harl-sokoban
   cd HARL
   python examples/train.py --algo happo --env sokoban --exp_name trial_happo --scenario TwoPlayer-Sokoban-v0
   ```

## What Agents Should Report

- Environment name
- Python version
- CUDA wheel channel
- Torch version
- Repositories cloned or reused
- Editable installs performed
- Verification commands run
- Final experiment launch commands

## Repository-Specific Note

For this Sokoban project, the current Linux setup helper is:
[setup_sokoban_linux.sh](/D:/讲义课件合集/多智能体基础/组队课题/MARL-Sokoban/MARL-project-sokoban/HARL/scripts/setup_sokoban_linux.sh:1)

It already follows the conda-first pattern and can be used as the default automation entrypoint for server setup.
