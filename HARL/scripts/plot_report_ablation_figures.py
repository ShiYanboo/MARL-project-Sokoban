#!/usr/bin/env python
"""Generate report-oriented ablation figures for Sokoban experiments."""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_sokoban_metrics import load_event_files, load_summary, smooth


M = 1_000_000

METRIC_TITLES = {
    "sokoban/train_success_rate": "train success rate",
    "sokoban/eval_success_rate": "eval success rate",
    "sokoban/train_box_completion_ratio": "train box completion ratio",
    "sokoban/eval_box_completion_ratio": "eval box completion ratio",
    "sokoban/train_mean_base_reward": "train base reward / step",
    "sokoban/eval_mean_base_reward": "eval base reward / step",
    "critic/value_loss": "critic value loss",
}


def seg(prefix, offset, contains=None):
    return {"prefix": prefix, "offset": offset, "contains": contains}


CHAINS = {
    "baseline": [
        seg("happo-baseline-original", 0),
        seg("happo-baseline-original-resume10m", 5 * M),
        seg("happo-baseline-original-resume15m", 10 * M),
        seg("happo-baseline-original-resume20m", 15 * M),
        seg("happo-baseline-original-mixv012-r1", 20 * M),
        seg("happo-baseline-original-mixv012-r2", 25 * M),
        seg("happo-baseline-original-mixv012-r3", 30 * M),
        seg("happo-baseline-original-mixv012-r4", 35 * M),
    ],
    "v1-nodl-nopush": [
        seg("happo-shaped-nodl-nopush", 0),
        seg("happo-shaped-nodl-nopush-resume5m", 5 * M),
        seg("happo-shaped-nodl-nopush-resume10m", 10 * M),
        seg("happo-shaped-nodl-nopush-resume15m", 15 * M),
        seg("happo-shaped-nodl-nopush-mixv012-r1", 20 * M),
        seg("happo-shaped-nodl-nopush-mixv012-r2", 25 * M),
        seg("happo-shaped-nodl-nopush-mixv012-r3", 30 * M),
        seg("happo-shaped-nodl-nopush-mixv012-r4", 35 * M),
    ],
    "v1-noshape": [
        seg("happo-shaped-noshape", 0),
        seg("happo-shaped-noshape-resume5m", 5 * M),
        seg("happo-shaped-noshape-resume10m", 10 * M),
        seg("happo-shaped-noshape-resume15m", 15 * M),
        seg("happo-shaped-noshape-mixv012-r1", 20 * M),
        seg("happo-shaped-noshape-mixv012-r2", 25 * M),
        seg("happo-shaped-noshape-mixv012-r3", 30 * M),
        seg("happo-shaped-noshape-mixv012-r4", 35 * M),
    ],
    "v1-strongpot": [
        seg("happo-shaped-strongpot", 0),
        seg("happo-shaped-strongpot-resume5m", 5 * M),
        seg("happo-shaped-strongpot-resume10m", 10 * M),
        seg("happo-shaped-strongpot-resume15m", 15 * M),
        seg("happo-shaped-strongpot-mixv012-r1", 20 * M),
        seg("happo-shaped-strongpot-mixv012-r2", 25 * M),
        seg("happo-shaped-strongpot-mixv012-r3", 30 * M),
        seg("happo-shaped-strongpot-mixv012-r4", 35 * M),
    ],
    "v2-plain": [
        seg("happo-shaped-v2", 0, "happo_turn_based_shaped_v2_conservative_finish200"),
    ],
    "v2-clip5": [
        seg("happo-shaped-v2", 0, "happo_turn_based_shaped_v2_conservative_cap5"),
    ],
    "v2-clip12": [
        seg("happo-shaped-v2", 0, "happo_turn_based_shaped_v2_conservative_cap12"),
        seg("happo-shaped-v2-cap12-resume10m", 5 * M),
        seg("happo-shaped-v2-cap12-resume15m", 10 * M),
        seg("happo-shaped-v2-cap12-resume20m", 15 * M),
        seg("happo-shaped-v2-cap12-mixv012-r1", 20 * M),
        seg("happo-shaped-v2-cap12-mixv012-r2", 25 * M),
        seg("happo-shaped-v2-cap12-mixv012-r3", 30 * M),
        seg("happo-shaped-v2-cap12-mixv012-r4", 35 * M),
    ],
    "cnn": [
        seg("happo-shaped-v2-cnn", 0, "happo_turn_based_cnn_shaped_v2_cap12_finish200"),
    ],
    "mlp8-action-history": [
        seg("happo-bfs-nodl-nopush-mlp-hist8", 0),
    ],
    "gru0": [
        seg("happo-rnn-bfs-nodl-nopush-nohist", 0),
        seg("happo-rnn-bfs-nodl-nopush-nohist-resume10m", 5 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-resume15m", 10 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-resume20m", 15 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1", 20 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2", 25 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r3", 30 * M),
        seg("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4", 35 * M),
    ],
    "gru1": [
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32", 0),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m", 5 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m", 10 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m", 15 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1", 20 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2", 25 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3", 30 * M),
        seg("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4", 35 * M),
    ],
    "gru8": [
        seg("happo-rnn-bfs-nodl-nopush", 0),
        seg("happo-rnn-bfs-nodl-nopush-resume10m", 5 * M),
        seg("happo-rnn-bfs-nodl-nopush-resume15m", 10 * M),
        seg("happo-rnn-bfs-nodl-nopush-resume20m", 15 * M),
        seg("happo-rnn-bfs-nodl-nopush-mixv012-r1", 20 * M),
        seg("happo-rnn-bfs-nodl-nopush-mixv012-r2", 25 * M),
        seg("happo-rnn-bfs-nodl-nopush-mixv012-r3", 30 * M),
        seg("happo-rnn-bfs-nodl-nopush-mixv012-r4", 35 * M),
    ],
    "hahypo-v1-alpha02": [
        seg("hahypo-a02-v1-nodl-nopush", 0),
        seg("hahypo-a02-v1-nodl-nopush-resume10m", 5 * M),
        seg("hahypo-a02-v1-nodl-nopush-resume15m", 10 * M),
        seg("hahypo-a02-v1-nodl-nopush-resume20m", 15 * M),
    ],
    "hahypo-v1-alpha05": [
        seg("hahypo-a05-v1-nodl-nopush", 0),
        seg("hahypo-a05-v1-nodl-nopush-resume10m", 5 * M),
        seg("hahypo-a05-v1-nodl-nopush-resume15m", 10 * M),
        seg("hahypo-a05-v1-nodl-nopush-resume20m", 15 * M),
    ],
    "hahypo-v1-alpha10": [
        seg("hahypo-a10-v1-nodl-nopush", 0),
        seg("hahypo-a10-v1-nodl-nopush-resume10m", 5 * M),
        seg("hahypo-a10-v1-nodl-nopush-resume15m", 10 * M),
        seg("hahypo-a10-v1-nodl-nopush-resume20m", 15 * M),
    ],
    "13tl-baseline": [
        seg("happo13tl-baseline-original", 0),
        seg("happo13tl-baseline-original-resume10m", 5 * M),
        seg("happo13tl-baseline-original-resume15m", 10 * M),
        seg("happo13tl-baseline-original-resume20m", 15 * M),
    ],
    "13tl-v1-nodl-nopush": [
        seg("happo13tl-v1-nodl-nopush", 0),
        seg("happo13tl-v1-nodl-nopush-resume10m", 5 * M),
        seg("happo13tl-v1-nodl-nopush-resume15m", 10 * M),
        seg("happo13tl-v1-nodl-nopush-resume20m", 15 * M),
    ],
    "13tl-gru0": [
        seg("happo13tl-gru-noaction", 0),
        seg("happo13tl-gru-noaction-resume10m", 5 * M),
        seg("happo13tl-gru-noaction-resume15m", 10 * M),
        seg("happo13tl-gru-noaction-resume20m", 15 * M),
    ],
    "13tl-cnn": [
        seg("happo13tl-cnn-nodl-nopush", 0),
        seg("happo13tl-cnn-nodl-nopush-resume10m", 5 * M),
        seg("happo13tl-cnn-nodl-nopush-resume15m", 10 * M),
        seg("happo13tl-cnn-nodl-nopush-resume20m", 15 * M),
    ],
    "13random-baseline": [
        seg("happo13-baseline-original", 0),
        seg("happo13-baseline-original-resume10m", 5 * M),
        seg("happo13-baseline-original-resume15m", 10 * M),
        seg("happo13-baseline-original-resume20m", 15 * M),
    ],
    "13random-v1-nodl-nopush": [
        seg("happo13-v1-nodl-nopush", 0),
        seg("happo13-v1-nodl-nopush-resume10m", 5 * M),
        seg("happo13-v1-nodl-nopush-resume15m", 10 * M),
        seg("happo13-v1-nodl-nopush-resume20m", 15 * M),
    ],
    "13random-gru0": [
        seg("happo13-gru-noaction", 0),
        seg("happo13-gru-noaction-resume10m", 5 * M),
        seg("happo13-gru-noaction-resume15m", 10 * M),
        seg("happo13-gru-noaction-resume20m", 15 * M),
    ],
    "13random-cnn": [
        seg("happo13-cnn-nodl-nopush", 0),
        seg("happo13-cnn-nodl-nopush-resume10m", 5 * M),
        seg("happo13-cnn-nodl-nopush-resume15m", 10 * M),
        seg("happo13-cnn-nodl-nopush-resume20m", 15 * M),
    ],
    "credit-progress005": [seg("happo-credit-progress005", 0)],
    "credit-progress010": [seg("happo-credit-progress010", 0)],
    "credit-reward002": [seg("happo-credit-reward002", 0)],
}


FIGURES = [
    {
        "filename": "01_main_train_success_box_ratio.png",
        "title": "7x7 v0 pretrain + pseudo-mix: main comparison",
        "families": {
            "baseline": "baseline",
            "v1-nopush-nodl": "v1-nodl-nopush",
            "v2-clip12": "v2-clip12",
            "gru0": "gru0",
            "gru1": "gru1",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/train_box_completion_ratio",
        ],
        "layout": (2, 1),
        "figsize": (16, 8),
        "boundary": 20 * M,
    },
    {
        "filename": "02_main_supp_train_eval_success_box_ratio.png",
        "title": "7x7 main comparison: train/eval success and box completion",
        "families": {
            "baseline": "baseline",
            "v1-nopush-nodl": "v1-nodl-nopush",
            "v2-clip12": "v2-clip12",
            "gru0": "gru0",
            "gru1": "gru1",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/eval_success_rate",
            "sokoban/train_box_completion_ratio",
            "sokoban/eval_box_completion_ratio",
        ],
        "layout": (4, 1),
        "figsize": (16, 13),
        "boundary": 20 * M,
    },
    {
        "filename": "03_main_supp_reward_value_loss.png",
        "title": "7x7 main comparison: base reward and critic loss",
        "families": {
            "baseline": "baseline",
            "v1-nopush-nodl": "v1-nodl-nopush",
            "v2-clip12": "v2-clip12",
            "gru0": "gru0",
            "gru1": "gru1",
        },
        "metrics": [
            "sokoban/train_mean_base_reward",
            "critic/value_loss",
        ],
        "layout": (2, 1),
        "figsize": (16, 8),
        "boundary": 20 * M,
    },
    {
        "filename": "04_v1_reward_ablation.png",
        "title": "7x7 v1 reward ablation",
        "families": {
            "v1-nopush-nodl": "v1-nodl-nopush",
            "v1-noshape": "v1-noshape",
            "v1": "v1-strongpot",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/train_box_completion_ratio",
        ],
        "layout": (2, 1),
        "figsize": (14, 7),
        "boundary": 20 * M,
    },
    {
        "filename": "05_v2_success_ablation.png",
        "title": "7x7 v2 distance cap ablation",
        "families": {
            "v2-no-clip": "v2-plain",
            "v2-clip12": "v2-clip12",
            "v2-clip5": "v2-clip5",
        },
        "metrics": ["sokoban/train_success_rate"],
        "layout": (1, 1),
        "figsize": (14, 4.2),
        "boundary": 20 * M,
    },
    {
        "filename": "06_v2_box_completion_ablation.png",
        "title": "7x7 v2 distance cap ablation: box completion",
        "families": {
            "v2-no-clip": "v2-plain",
            "v2-clip12": "v2-clip12",
            "v2-clip5": "v2-clip5",
        },
        "metrics": ["sokoban/train_box_completion_ratio"],
        "layout": (1, 1),
        "figsize": (14, 4.2),
        "boundary": 20 * M,
    },
    {
        "filename": "07_architecture_train_metrics.png",
        "title": "7x7 architecture comparison",
        "families": {
            "mlp v1-nopush-nodl": "v1-nodl-nopush",
            "cnn": "cnn",
            "gru0": "gru0",
            "gru1": "gru1",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/train_box_completion_ratio",
        ],
        "layout": (2, 1),
        "figsize": (16, 8),
        "boundary": 20 * M,
    },
    {
        "filename": "08_architecture_supp_eval_value_reward.png",
        "title": "7x7 architecture comparison: supplementary metrics",
        "families": {
            "mlp v1-nopush-nodl": "v1-nodl-nopush",
            "cnn": "cnn",
            "gru0": "gru0",
            "gru1": "gru1",
        },
        "metrics": [
            "sokoban/eval_box_completion_ratio",
            "critic/value_loss",
            "sokoban/train_mean_base_reward",
        ],
        "layout": (3, 1),
        "figsize": (16, 11),
        "boundary": 20 * M,
    },
    {
        "filename": "09_action_history_ablation.png",
        "title": "Action-history ablation",
        "panels": [
            {
                "title": "GRU history length",
                "families": {"gru-0": "gru0", "gru-1": "gru1", "gru-8": "gru8"},
                "metric": "sokoban/train_success_rate",
            },
            {
                "title": "MLP-8 vs GRU-8",
                "families": {"mlp-8": "mlp8-action-history", "gru-8": "gru8"},
                "metric": "sokoban/train_success_rate",
            },
        ],
        "figsize": (14, 7),
        "boundary": 20 * M,
    },
    {
        "filename": "10_hahypo_vs_happo_v1.png",
        "title": "HAHyPO vs HAPPO on v1-nopush-nodl",
        "families": {
            "HAPPO alpha=0.0": "v1-nodl-nopush",
            "HAHyPO alpha=0.2": "hahypo-v1-alpha02",
            "HAHyPO alpha=0.5": "hahypo-v1-alpha05",
            "HAHyPO alpha=1.0": "hahypo-v1-alpha10",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/train_box_completion_ratio",
        ],
        "layout": (2, 1),
        "figsize": (16, 8),
    },
    {
        "filename": "12_13tl_box_completion.png",
        "title": "13x13 top-left padding: box completion",
        "families": {
            "v1-nopush-nodl": "13tl-v1-nodl-nopush",
            "gru0": "13tl-gru0",
            "baseline": "13tl-baseline",
            "cnn": "13tl-cnn",
        },
        "metrics": ["sokoban/train_box_completion_ratio"],
        "layout": (1, 1),
        "figsize": (14, 4.2),
    },
    {
        "filename": "13_13tl_supp_metrics.png",
        "title": "13x13 top-left padding: supplementary metrics",
        "families": {
            "v1-nopush-nodl": "13tl-v1-nodl-nopush",
            "gru0": "13tl-gru0",
            "baseline": "13tl-baseline",
            "cnn": "13tl-cnn",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "critic/value_loss",
            "sokoban/eval_success_rate",
            "sokoban/eval_box_completion_ratio",
        ],
        "layout": (2, 2),
        "figsize": (15, 8.5),
    },
    {
        "filename": "14_13random_train_metrics.png",
        "title": "13x13 random padding",
        "families": {
            "gru0": "13random-gru0",
            "baseline": "13random-baseline",
            "v1": "13random-v1-nodl-nopush",
            "cnn": "13random-cnn",
        },
        "metrics": [
            "sokoban/train_success_rate",
            "sokoban/train_box_completion_ratio",
        ],
        "layout": (2, 1),
        "figsize": (14, 7),
    },
    {
        "filename": "15_13random_value_reward.png",
        "title": "13x13 random padding: value and reward",
        "families": {
            "gru0": "13random-gru0",
            "baseline": "13random-baseline",
            "v1": "13random-v1-nodl-nopush",
            "cnn": "13random-cnn",
        },
        "metrics": [
            "critic/value_loss",
            "sokoban/train_mean_base_reward",
        ],
        "layout": (1, 2),
        "figsize": (15, 4.8),
    },
    {
        "filename": "16_credit_assignment.png",
        "title": "Credit assignment ablation",
        "families": {
            "v1-nopush-nodl": "v1-nodl-nopush",
            "+progress005": "credit-progress005",
            "+progress010": "credit-progress010",
            "+reward002": "credit-reward002",
        },
        "metrics": [
            "sokoban/train_box_completion_ratio",
            "sokoban/train_success_rate",
        ],
        "layout": (2, 1),
        "figsize": (14, 7),
    },
]


def latest_run_dir(results_root, prefix, contains=None):
    prefix_dir = results_root / prefix
    if not prefix_dir.exists():
        return None
    candidates = []
    for marker in ("progress.txt", "config.json"):
        for marker_path in prefix_dir.rglob(marker):
            run_dir = marker_path.parent if marker == "progress.txt" else marker_path.parent
            if contains and contains not in str(run_dir):
                continue
            if run_dir not in candidates:
                candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def max_logged_step(run_dir):
    progress_path = run_dir / "progress.txt"
    max_step = 0
    if not progress_path.exists():
        return max_step
    for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in [r"total num timesteps\s+(\d+)/(\d+)", r"^\s*(\d+(?:\.\d+)?)\s*,"]:
            match = re.search(pattern, line)
            if match:
                max_step = max(max_step, int(float(match.group(1))))
    return max_step


def target_steps(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config["algo_args"]["train"]["num_env_steps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_progress_reward(run_dir):
    progress_path = run_dir / "progress.txt"
    if not progress_path.exists():
        return []
    points = []
    for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            points.append((int(float(parts[0])), float(parts[1])))
        except ValueError:
            continue
    return points


@lru_cache(maxsize=None)
def load_run_metrics(run_dir_string):
    run_dir = Path(run_dir_string)
    metrics = {}
    summary_path = run_dir / "logs" / "summary.json"
    if summary_path.exists():
        metrics.update(load_summary(summary_path))
    log_dir = run_dir / "logs"
    if log_dir.exists():
        try:
            for key, points in load_event_files(log_dir).items():
                metrics.setdefault(key, points)
        except (ImportError, FileNotFoundError, RuntimeError):
            pass
    if "train_episode_rewards/aver_rewards" not in metrics:
        progress_points = load_progress_reward(run_dir)
        if progress_points:
            metrics["train_episode_rewards/aver_rewards"] = progress_points
    return metrics


def chain_series(results_root, chain, metrics):
    by_metric = {metric: [] for metric in metrics}
    index_entries = []
    for segment in chain:
        run_dir = latest_run_dir(results_root, segment["prefix"], segment.get("contains"))
        if run_dir is None:
            index_entries.append({**segment, "status": "missing", "run_dir": None})
            continue
        logged = max_logged_step(run_dir)
        target = target_steps(run_dir)
        status = "partial"
        if target and logged >= int(target * 0.95):
            status = "complete"
        index_entries.append(
            {
                **segment,
                "status": status,
                "run_dir": str(run_dir),
                "logged_step": logged,
                "target": target,
            }
        )
        run_metrics = load_run_metrics(str(run_dir))
        for metric in metrics:
            for step, value in run_metrics.get(metric, []):
                by_metric[metric].append((segment["offset"] + step, value))
    return {metric: sorted(points) for metric, points in by_metric.items()}, index_entries


def plot_panel(axis, results_root, families, metric, smoothing, max_step=None, boundary=None):
    plotted = False
    used = {}
    for label, chain_name in families.items():
        points_by_metric, entries = chain_series(results_root, CHAINS[chain_name], [metric])
        used[label] = entries
        points = points_by_metric.get(metric, [])
        if max_step is not None:
            points = [(step, value) for step, value in points if step <= max_step]
        if not points:
            continue
        steps = [step / M for step, _ in points]
        values = [value for _, value in points]
        axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=label)
        plotted = True
    if boundary is not None:
        axis.axvline(boundary / M, color="black", alpha=0.25, linestyle="--", linewidth=1.2)
    axis.set_title(METRIC_TITLES.get(metric, metric))
    axis.set_xlabel("stitched env steps (M)")
    axis.grid(alpha=0.3)
    if plotted:
        axis.legend(fontsize=8)
    return plotted, used


def plot_standard_figure(results_root, figure, output_path, smoothing):
    rows, cols = figure["layout"]
    fig, axes = plt.subplots(rows, cols, figsize=figure["figsize"], squeeze=False)
    all_used = {}
    any_plotted = False
    for axis, metric in zip(axes.flat, figure["metrics"]):
        plotted, used = plot_panel(
            axis,
            results_root,
            figure["families"],
            metric,
            smoothing,
            max_step=figure.get("max_step"),
            boundary=figure.get("boundary"),
        )
        any_plotted = any_plotted or plotted
        for label, entries in used.items():
            all_used.setdefault(label, entries)
    for axis in axes.flat[len(figure["metrics"]):]:
        axis.axis("off")
    fig.suptitle(figure["title"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if any_plotted:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return any_plotted, all_used


def shortest_max_step(results_root, panels):
    max_steps = []
    for panel in panels:
        for chain_name in panel["families"].values():
            points_by_metric, _ = chain_series(results_root, CHAINS[chain_name], [panel["metric"]])
            points = points_by_metric.get(panel["metric"], [])
            if points:
                max_steps.append(max(step for step, _ in points))
    return min(max_steps) if max_steps else None


def plot_multi_panel_figure(results_root, figure, output_path, smoothing):
    panels = figure["panels"]
    fig, axes = plt.subplots(len(panels), 1, figsize=figure["figsize"], squeeze=False)
    all_used = {}
    any_plotted = False
    common_max = shortest_max_step(results_root, panels)
    for axis, panel in zip(axes.flat, panels):
        plotted, used = plot_panel(
            axis,
            results_root,
            panel["families"],
            panel["metric"],
            smoothing,
            max_step=common_max,
            boundary=figure.get("boundary"),
        )
        axis.set_title(f"{panel['title']}: {METRIC_TITLES.get(panel['metric'], panel['metric'])}")
        any_plotted = any_plotted or plotted
        for label, entries in used.items():
            all_used.setdefault(f"{panel['title']} / {label}", entries)
    fig.suptitle(figure["title"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if any_plotted:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return any_plotted, all_used


def write_index(output_dir, generated, used_by_figure):
    lines = [
        "# Report Ablation Figure Index",
        "",
        "Figure labels use `clip5/clip12` for report wording, but these correspond to the earlier distance-cap ablations.",
        "Missing or partial segments are recorded below; figures use whatever scalar event data is present.",
        "",
        "Skipped: requested HAHyPO architecture comparison with `gru0 + alpha=0.2` is not generated because run17 HAHyPO GRU experiments used GRU-8/action-history-8, not GRU-0.",
        "Skipped: useless-action figure is intentionally left out until those long runs produce enough data.",
        "",
        "## Generated Figures",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.append("")
    lines.append("## Run Sources")
    for figure_name, used in used_by_figure.items():
        lines.append(f"### {figure_name}")
        if not used:
            lines.append("- no plotted data")
            continue
        for label, entries in used.items():
            lines.append(f"- {label}")
            for entry in entries:
                prefix = entry["prefix"]
                contains = entry.get("contains")
                suffix = f", contains `{contains}`" if contains else ""
                if entry["status"] == "missing":
                    lines.append(f"  - `{prefix}`{suffix}: missing")
                else:
                    lines.append(
                        f"  - `{prefix}`{suffix}: {entry['status']}, "
                        f"local {entry.get('logged_step')}/{entry.get('target')}, "
                        f"offset {entry['offset']}, run `{entry['run_dir']}`"
                    )
        lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/report-ablation-figures"))
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    used_by_figure = {}
    for figure in FIGURES:
        output_path = args.output_dir / figure["filename"]
        if "panels" in figure:
            plotted, used = plot_multi_panel_figure(args.results_root, figure, output_path, args.smoothing)
        else:
            plotted, used = plot_standard_figure(args.results_root, figure, output_path, args.smoothing)
        used_by_figure[figure["filename"]] = used
        if plotted:
            generated.append(output_path)
            print(f"Saved plot: {output_path}")
        else:
            print(f"No data for: {output_path}")
    write_index(args.output_dir, generated, used_by_figure)
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
