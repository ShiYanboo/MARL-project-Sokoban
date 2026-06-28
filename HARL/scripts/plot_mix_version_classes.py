#!/usr/bin/env python
"""Classify and plot Sokoban runs whose names mention mix012."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sokoban_metrics import load_event_files, load_summary, smooth


METRICS = [
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/eval_final_boxes_on_target", "eval boxes on target"),
    ("sokoban/train_success_rate", "train success rate"),
    ("sokoban/train_box_completion_ratio", "train box completion ratio"),
    ("eval_average_episode_rewards", "eval episode reward"),
    ("train_episode_rewards/aver_rewards", "train episode reward"),
    ("critic/value_loss", "critic value loss"),
]


PSEUDO_MIX_FULL = {
    "v1 strongpot pseudo-mix": [
        ("happo-shaped-strongpot-mixv012-r1", 0),
        ("happo-shaped-strongpot-mixv012-r2", 5_000_000),
        ("happo-shaped-strongpot-mixv012-r3", 10_000_000),
        ("happo-shaped-strongpot-mixv012-r4", 15_000_000),
    ],
    "v1 nodeadlock pseudo-mix": [
        ("happo-shaped-nodeadlock-mixv012-r1", 0),
        ("happo-shaped-nodeadlock-mixv012-r2", 5_000_000),
        ("happo-shaped-nodeadlock-mixv012-r3", 10_000_000),
        ("happo-shaped-nodeadlock-mixv012-r4", 15_000_000),
    ],
    "v1 nodl-nopush pseudo-mix": [
        ("happo-shaped-nodl-nopush-mixv012-r1", 0),
        ("happo-shaped-nodl-nopush-mixv012-r2", 5_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r3", 10_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r4", 15_000_000),
    ],
    "v1 noshape pseudo-mix": [
        ("happo-shaped-noshape-mixv012-r1", 0),
        ("happo-shaped-noshape-mixv012-r2", 5_000_000),
        ("happo-shaped-noshape-mixv012-r3", 10_000_000),
        ("happo-shaped-noshape-mixv012-r4", 15_000_000),
    ],
}


PSEUDO_MIX_PARTIAL_COMPLETED_ROUNDS = {
    "v2 cap12 pseudo-mix completed rounds": [
        ("happo-shaped-v2-cap12-mixv012-r1", 0),
        ("happo-shaped-v2-cap12-mixv012-r2", 5_000_000),
        ("happo-shaped-v2-cap12-mixv012-r3", 10_000_000),
        ("happo-shaped-v2-cap12-mixv012-r4", 15_000_000),
    ],
    "gru hist8 pseudo-mix completed rounds": [
        ("happo-rnn-bfs-nodl-nopush-mixv012-r1", 0),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r2", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r3", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r4", 15_000_000),
    ],
}


TRUE_MIX_PADDED_STARTED = {
    "13 random gru noaction true-mix": [
        ("happo13-gru-noaction-mixv012-r1", 0),
    ],
    "13 random gru 8action true-mix": [
        ("happo13-gru-8action-mixv012-r1", 0),
    ],
    "13 random gru 1action true-mix": [
        ("happo13-gru-1action-mixv012-r1", 0),
    ],
}


def latest_run_dir(results_root, result_prefix):
    prefix_dir = results_root / result_prefix
    candidates = [path.parent for path in prefix_dir.rglob("config.json")]
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
    except Exception:
        return None


def classify_run(run_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env = config["env_args"]
    pool = env.get("scenario_pool")
    dim_room = env.get("dim_room")
    num_boxes = env.get("num_boxes")
    padded = bool(env.get("use_mixed_obs_padding", False))
    obs_dim = env.get("obs_dim_room")
    padding_mode = env.get("mixed_obs_padding_mode")
    if pool and dim_room is not None:
        label = "pseudo_mix_fixed_size"
    elif pool and padded and obs_dim:
        label = f"true_mix_padded_{padding_mode}"
    elif pool:
        label = "true_mix_unpadded_or_unknown"
    elif padded:
        label = f"padded_single_{padding_mode}"
    else:
        label = "fixed_single"
    return {
        "label": label,
        "scenario_pool": pool,
        "dim_room": dim_room,
        "num_boxes": num_boxes,
        "use_mixed_obs_padding": padded,
        "obs_dim_room": obs_dim,
        "mixed_obs_padding_mode": padding_mode,
    }


def is_complete(run_dir, threshold=0.95):
    target = target_steps(run_dir)
    return bool(target and max_logged_step(run_dir) >= threshold * target)


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


def load_metrics(run_dir):
    metrics = {}
    summary_path = run_dir / "logs" / "summary.json"
    if summary_path.exists():
        metrics.update(load_summary(summary_path))
    log_dir = run_dir / "logs"
    if log_dir.exists():
        try:
            metrics.update(load_event_files(log_dir))
        except (ImportError, FileNotFoundError, RuntimeError):
            pass
    if "train_episode_rewards/aver_rewards" not in metrics:
        progress_points = load_progress_reward(run_dir)
        if progress_points:
            metrics["train_episode_rewards/aver_rewards"] = progress_points
    return metrics


def collect(results_root, families, require_complete):
    all_series = {}
    used = {}
    for family_name, segments in families.items():
        series = {}
        family_used = []
        for metric_key, _ in METRICS:
            points = []
            for prefix, offset in segments:
                run_dir = latest_run_dir(results_root, prefix)
                if run_dir is None:
                    if metric_key == METRICS[0][0]:
                        family_used.append((prefix, None, "missing", None))
                    continue
                complete = is_complete(run_dir)
                if metric_key == METRICS[0][0]:
                    family_used.append((prefix, run_dir, "complete" if complete else "partial", classify_run(run_dir)))
                if require_complete and not complete:
                    continue
                metric_points = load_metrics(run_dir).get(metric_key, [])
                points.extend((step + offset, value) for step, value in metric_points)
            series[metric_key] = sorted(points)
        all_series[family_name] = series
        used[family_name] = family_used
    return all_series, used


def plot_series(all_series, output_path, title, smoothing):
    available = [
        (metric_key, metric_title)
        for metric_key, metric_title in METRICS
        if any(series.get(metric_key) for series in all_series.values())
    ]
    if not available:
        return False
    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for axis, (metric_key, metric_title) in zip(axes.flat, available):
        for family_name, series in all_series.items():
            points = series.get(metric_key, [])
            if not points:
                continue
            steps = [step for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=family_name)
        axis.set_title(metric_title)
        axis.set_xlabel("env steps stitched within this class")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)
    for axis in axes.flat[len(available):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_index(output_dir, used_groups):
    lines = [
        "# Mix Version Classification",
        "",
        "Definitions:",
        "- `pseudo_mix_fixed_size`: `scenario_pool` is set, but `dim_room`/`num_boxes` are also forced. This is not a true size/difficulty mix.",
        "- `true_mix_padded_*`: `scenario_pool` is set, fixed `dim_room`/`num_boxes` are absent, and fixed-size padded observations are enabled.",
        "",
    ]
    for group_name, used in used_groups.items():
        lines.append(f"## {group_name}")
        for family_name, runs in used.items():
            lines.append(f"### {family_name}")
            for prefix, run_dir, status, cfg in runs:
                if run_dir is None:
                    lines.append(f"- {prefix}: missing")
                    continue
                step = max_logged_step(run_dir)
                target = target_steps(run_dir)
                lines.append(
                    f"- {prefix}: {status}, step {step}/{target}, class `{cfg['label']}`, "
                    f"pool `{cfg['scenario_pool']}`, dim `{cfg['dim_room']}`, boxes `{cfg['num_boxes']}`, "
                    f"pad `{cfg['use_mixed_obs_padding']}`, obs_dim `{cfg['obs_dim_room']}`, "
                    f"padding `{cfg['mixed_obs_padding_mode']}`, run `{run_dir}`"
                )
            lines.append("")
    (output_dir / "mix_version_index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/mix-version-classes"))
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full, used_full = collect(args.results_root, PSEUDO_MIX_FULL, require_complete=True)
    partial, used_partial = collect(
        args.results_root, PSEUDO_MIX_PARTIAL_COMPLETED_ROUNDS, require_complete=True
    )
    true_started, used_true = collect(
        args.results_root, TRUE_MIX_PADDED_STARTED, require_complete=False
    )

    generated = []
    if plot_series(
        full,
        output_dir / "pseudo_mix_fixed_size_full_20m.png",
        "Pseudo-mix fixed-size runs: completed full 20M workflows",
        args.smoothing,
    ):
        generated.append(output_dir / "pseudo_mix_fixed_size_full_20m.png")
    if plot_series(
        partial,
        output_dir / "pseudo_mix_fixed_size_completed_rounds_partial_families.png",
        "Pseudo-mix fixed-size runs: completed rounds only for partial families",
        args.smoothing,
    ):
        generated.append(output_dir / "pseudo_mix_fixed_size_completed_rounds_partial_families.png")
    if plot_series(
        true_started,
        output_dir / "true_mix_padded_started.png",
        "True padded mix runs: currently started only",
        args.smoothing,
    ):
        generated.append(output_dir / "true_mix_padded_started.png")

    write_index(
        output_dir,
        {
            "pseudo mix fixed-size full 20m": used_full,
            "pseudo mix fixed-size partial families": used_partial,
            "true padded mix started": used_true,
        },
    )
    print(f"Saved plots to: {output_dir}")
    for path in generated:
        print(f"  - {path}")
    print(f"Index: {output_dir / 'mix_version_index.md'}")


if __name__ == "__main__":
    main()
