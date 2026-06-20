---
name: conda-experiment-workflow
description: Standard conda-first workflow for reproducible ML and research workspaces. Use when Codex needs to prepare a fresh environment, install baseline packages, clone one or more repositories, perform editable local installs, verify CUDA or PyTorch, and then launch or document experiments on Linux servers or local development machines.
---

# Conda Experiment Workflow

Use this skill when the task is not just "run one command", but "prepare a clean experiment workspace that should stay reproducible".

Read [references/command-templates.md](references/command-templates.md) when you need concrete shell snippets for conda creation, PyTorch CUDA wheels, editable installs, cloning, or experiment launch.

## Default Rules

- Do not install experiment dependencies into `base`.
- Prefer a named conda environment. Default to `project-slug` or `project-slug-cu118`.
- Prefer Python `3.10` unless the repository clearly requires another version.
- Install PyTorch before the rest of the ML stack when CUDA matters.
- Use explicit CUDA wheel channels such as `cu118`, `cu121`, `cu124`, or `cu126`.
- Prefer `pip install -e <repo>` for local repositories that will be edited.
- Verify imports, CUDA visibility, and one minimal run before declaring setup complete.
- Record the exact environment name, torch build, and launch command in the final notes or docs.

## Workflow

1. Inspect the workspace.
   Confirm whether the repo already exists, whether a conda env already exists, which branch is active, and whether CUDA must be supported.

2. Choose or confirm the environment name.
   Use a stable, descriptive name such as `harl-sokoban`, `project-slug-cu118`, or `paper-repro-2026`.

3. Create or reuse the conda environment.
   Create it only if it does not already exist. Avoid deleting or replacing an environment unless the user asked for that explicitly.

4. Install packaging basics.
   Upgrade `pip`, `wheel`, and `setuptools` inside the target environment before installing project dependencies.

5. Install core framework packages.
   If CUDA matters, install torch-family packages from the explicit PyTorch index URL that matches the chosen CUDA channel.

6. Clone or update repositories.
   Clone only the repos needed for the task into the intended workspace root. Do not scatter clones across unrelated directories.

7. Install local repositories in editable mode.
   Use editable installs for repos that may be patched, extended, or actively developed during the task.

8. Verify the setup.
   Run targeted checks: import checks, `torch.cuda.is_available()`, version prints, and one minimal training or validation command.

9. Launch experiments from the named environment.
   Prefer `conda run -n <env> ...` in automation and `conda activate <env>` for interactive sessions.

10. Leave a reproducible trail.
    Write down the environment name, CUDA channel, key package versions, cloned repos, and canonical launch commands.

## Decision Rules

- If the repo already ships a setup script, inspect it before inventing a new one.
- If the user has both Windows and Linux targets, prepare Linux-first experiment instructions and keep Windows validation separate.
- If the machine is a shared server, avoid `sudo` unless the user explicitly asked for system-level changes.
- If package versions are uncertain, prefer the repo's pinned versions over newer ones.
- If PyTorch CUDA compatibility is unclear, check `nvidia-smi` and choose the most conservative supported wheel channel first.

## Deliverables

- A named conda environment
- A repeatable install script or command block
- A short verification record
- Canonical experiment launch commands
- Notes about CUDA channel, torch build, and any repo-specific caveats

## Output Style

- Keep setup notes concrete.
- Prefer exact commands over vague advice.
- Mention environment names explicitly.
- Separate one-time setup commands from repeated experiment commands.
