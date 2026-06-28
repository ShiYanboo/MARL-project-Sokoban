#!/usr/bin/env python
"""Plot stitched Sokoban pretrain and mix curves for selected experiment families."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sokoban_metrics import load_event_files, load_summary, smooth


METRICS = [
    ("sokoban/train_success_rate", "train success rate"),
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/train_box_completion_ratio", "train box completion ratio"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/train_mean_base_reward", "train base reward / step"),
    ("sokoban/eval_mean_base_reward", "eval base reward / step"),
    ("critic/value_loss", "critic value loss"),
    ("eval_average_episode_rewards", "eval episode reward"),
]


OLD_7X7 = {
    "baseline original": [
        ("happo-baseline-original", 0),
        ("happo-baseline-original-resume10m", 5_000_000),
        ("happo-baseline-original-resume15m", 10_000_000),
        ("happo-baseline-original-resume20m", 15_000_000),
        ("happo-baseline-original-mixv012-r1", 20_000_000),
        ("happo-baseline-original-mixv012-r2", 25_000_000),
        ("happo-baseline-original-mixv012-r3", 30_000_000),
        ("happo-baseline-original-mixv012-r4", 35_000_000),
    ],
    "v1 nodl-nopush": [
        ("happo-shaped-nodl-nopush", 0),
        ("happo-shaped-nodl-nopush-resume5m", 5_000_000),
        ("happo-shaped-nodl-nopush-resume10m", 10_000_000),
        ("happo-shaped-nodl-nopush-resume15m", 15_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r4", 35_000_000),
    ],
    "gru0 no-action-history": [
        ("happo-rnn-bfs-nodl-nopush-nohist", 0),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume20m", 15_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1", 20_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2", 25_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r3", 30_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4", 35_000_000),
    ],
    "gru1 one-action-history": [
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32", 0),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m", 15_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1", 20_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2", 25_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3", 30_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4", 35_000_000),
    ],
    "gru8 eight-action-history": [
        ("happo-rnn-bfs-nodl-nopush", 0),
        ("happo-rnn-bfs-nodl-nopush-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-resume20m", 15_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r4", 35_000_000),
    ],
}


LATEST_13_TOP_LEFT = {
    "13tl baseline original": [
        ("happo13tl-baseline-original", 0),
        ("happo13tl-baseline-original-resume10m", 5_000_000),
        ("happo13tl-baseline-original-resume15m", 10_000_000),
        ("happo13tl-baseline-original-resume20m", 15_000_000),
        ("happo13tl-baseline-original-mixv012-r1", 20_000_000),
        ("happo13tl-baseline-original-mixv012-r2", 25_000_000),
        ("happo13tl-baseline-original-mixv012-r3", 30_000_000),
        ("happo13tl-baseline-original-mixv012-r4", 35_000_000),
    ],
    "13tl v1 nodl-nopush": [
        ("happo13tl-v1-nodl-nopush", 0),
        ("happo13tl-v1-nodl-nopush-resume10m", 5_000_000),
        ("happo13tl-v1-nodl-nopush-resume15m", 10_000_000),
        ("happo13tl-v1-nodl-nopush-resume20m", 15_000_000),
        ("happo13tl-v1-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo13tl-v1-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo13tl-v1-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo13tl-v1-nodl-nopush-mixv012-r4", 35_000_000),
    ],
    "13tl gru0 no-action-history": [
        ("happo13tl-gru-noaction", 0),
        ("happo13tl-gru-noaction-resume10m", 5_000_000),
        ("happo13tl-gru-noaction-resume15m", 10_000_000),
        ("happo13tl-gru-noaction-resume20m", 15_000_000),
        ("happo13tl-gru-noaction-mixv012-r1", 20_000_000),
        ("happo13tl-gru-noaction-mixv012-r2", 25_000_000),
        ("happo13tl-gru-noaction-mixv012-r3", 30_000_000),
        ("happo13tl-gru-noaction-mixv012-r4", 35_000_000),
    ],
    "13tl gru1 one-action-history": [
        ("happo13tl-gru-1action", 0),
        ("happo13tl-gru-1action-resume10m", 5_000_000),
        ("happo13tl-gru-1action-resume15m", 10_000_000),
        ("happo13tl-gru-1action-resume20m", 15_000_000),
        ("happo13tl-gru-1action-mixv012-r1", 20_000_000),
        ("happo13tl-gru-1action-mixv012-r2", 25_000_000),
        ("happo13tl-gru-1action-mixv012-r3", 30_000_000),
        ("happo13tl-gru-1action-mixv012-r4", 35_000_000),
    ],
    "13tl gru8 eight-action-history": [
        ("happo13tl-gru-8action", 0),
        ("happo13tl-gru-8action-resume10m", 5_000_000),
        ("happo13tl-gru-8action-resume15m", 10_000_000),
        ("happo13tl-gru-8action-resume20m", 15_000_000),
        ("happo13tl-gru-8action-mixv012-r1", 20_000_000),
        ("happo13tl-gru-8action-mixv012-r2", 25_000_000),
        ("happo13tl-gru-8action-mixv012-r3", 30_000_000),
        ("happo13tl-gru-8action-mixv012-r4", 35_000_000),
    ],
}


RANDOM_13 = {
    "13 random baseline original": [
        ("happo13-baseline-original", 0),
        ("happo13-baseline-original-resume10m", 5_000_000),
        ("happo13-baseline-original-resume15m", 10_000_000),
        ("happo13-baseline-original-resume20m", 15_000_000),
        ("happo13-baseline-original-mixv012-r1", 20_000_000),
        ("happo13-baseline-original-mixv012-r2", 25_000_000),
        ("happo13-baseline-original-mixv012-r3", 30_000_000),
        ("happo13-baseline-original-mixv012-r4", 35_000_000),
    ],
    "13 random v1 noshape": [
        ("happo13-v1-noshape", 0),
        ("happo13-v1-noshape-resume10m", 5_000_000),
        ("happo13-v1-noshape-resume15m", 10_000_000),
        ("happo13-v1-noshape-resume20m", 15_000_000),
        ("happo13-v1-noshape-mixv012-r1", 20_000_000),
        ("happo13-v1-noshape-mixv012-r2", 25_000_000),
        ("happo13-v1-noshape-mixv012-r3", 30_000_000),
        ("happo13-v1-noshape-mixv012-r4", 35_000_000),
    ],
    "13 random v1 nodl-nopush": [
        ("happo13-v1-nodl-nopush", 0),
        ("happo13-v1-nodl-nopush-resume10m", 5_000_000),
        ("happo13-v1-nodl-nopush-resume15m", 10_000_000),
        ("happo13-v1-nodl-nopush-resume20m", 15_000_000),
        ("happo13-v1-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo13-v1-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo13-v1-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo13-v1-nodl-nopush-mixv012-r4", 35_000_000),
    ],
    "13 random gru0 no-action-history": [
        ("happo13-gru-noaction", 0),
        ("happo13-gru-noaction-resume10m", 5_000_000),
        ("happo13-gru-noaction-resume15m", 10_000_000),
        ("happo13-gru-noaction-resume20m", 15_000_000),
        ("happo13-gru-noaction-mixv012-r1", 20_000_000),
        ("happo13-gru-noaction-mixv012-r2", 25_000_000),
        ("happo13-gru-noaction-mixv012-r3", 30_000_000),
        ("happo13-gru-noaction-mixv012-r4", 35_000_000),
    ],
    "13 random gru1 one-action-history": [
        ("happo13-gru-1action", 0),
        ("happo13-gru-1action-resume10m", 5_000_000),
        ("happo13-gru-1action-resume15m", 10_000_000),
        ("happo13-gru-1action-resume20m", 15_000_000),
        ("happo13-gru-1action-mixv012-r1", 20_000_000),
        ("happo13-gru-1action-mixv012-r2", 25_000_000),
        ("happo13-gru-1action-mixv012-r3", 30_000_000),
        ("happo13-gru-1action-mixv012-r4", 35_000_000),
    ],
    "13 random gru8 eight-action-history": [
        ("happo13-gru-8action", 0),
        ("happo13-gru-8action-resume10m", 5_000_000),
        ("happo13-gru-8action-resume15m", 10_000_000),
        ("happo13-gru-8action-resume20m", 15_000_000),
        ("happo13-gru-8action-mixv012-r1", 20_000_000),
        ("happo13-gru-8action-mixv012-r2", 25_000_000),
        ("happo13-gru-8action-mixv012-r3", 30_000_000),
        ("happo13-gru-8action-mixv012-r4", 35_000_000),
    ],
    "13 random cnn nodl-nopush": [
        ("happo13-cnn-nodl-nopush", 0),
        ("happo13-cnn-nodl-nopush-resume10m", 5_000_000),
        ("happo13-cnn-nodl-nopush-resume15m", 10_000_000),
        ("happo13-cnn-nodl-nopush-resume20m", 15_000_000),
        ("happo13-cnn-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo13-cnn-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo13-cnn-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo13-cnn-nodl-nopush-mixv012-r4", 35_000_000),
    ],
}


def latest_run_dir(results_root, result_prefix):
    prefix_dir = results_root / result_prefix
    if not prefix_dir.exists():
        return None
    candidates = [path.parent for path in prefix_dir.rglob("progress.txt")]
    if not candidates:
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
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def classify_run(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {"label": "unknown"}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"label": "unknown"}
    env = config.get("env_args", {})
    pool = env.get("scenario_pool")
    dim_room = env.get("dim_room")
    num_boxes = env.get("num_boxes")
    padded = bool(env.get("use_mixed_obs_padding", False))
    padding_mode = env.get("mixed_obs_padding_mode")
    obs_dim = env.get("obs_dim_room")
    if pool and dim_room is not None:
        label = "pseudo_mix_fixed_size"
    elif pool and padded:
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


def collect(results_root, families):
    all_series = {}
    index = {}
    for family_name, segments in families.items():
        metrics_by_key = {metric_key: [] for metric_key, _ in METRICS}
        family_index = []
        for prefix, offset in segments:
            run_dir = latest_run_dir(results_root, prefix)
            if run_dir is None:
                family_index.append(
                    {
                        "prefix": prefix,
                        "offset": offset,
                        "status": "missing",
                        "run_dir": None,
                    }
                )
                continue

            logged_step = max_logged_step(run_dir)
            target = target_steps(run_dir)
            status = "partial"
            if target and logged_step >= int(target * 0.95):
                status = "complete"
            family_index.append(
                {
                    "prefix": prefix,
                    "offset": offset,
                    "status": status,
                    "logged_step": logged_step,
                    "target": target,
                    "run_dir": str(run_dir),
                    "class": classify_run(run_dir),
                }
            )

            run_metrics = load_metrics(run_dir)
            for metric_key, _ in METRICS:
                for step, value in run_metrics.get(metric_key, []):
                    metrics_by_key[metric_key].append((offset + step, value))

        all_series[family_name] = {
            metric_key: sorted(points) for metric_key, points in metrics_by_key.items()
        }
        index[family_name] = family_index
    return all_series, index


def plot_series(all_series, output_path, title, smoothing, mix_boundary=20_000_000):
    available = [
        (metric_key, metric_title)
        for metric_key, metric_title in METRICS
        if any(series.get(metric_key) for series in all_series.values())
    ]
    if not available:
        return False

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.6 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for axis, (metric_key, metric_title) in zip(axes.flat, available):
        for family_name, series in all_series.items():
            points = series.get(metric_key, [])
            if not points:
                continue
            steps = [step / 1_000_000 for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=family_name)
        axis.axvline(mix_boundary / 1_000_000, color="black", alpha=0.25, linestyle="--")
        axis.set_title(metric_title)
        axis.set_xlabel("stitched env steps (M)")
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


def write_index(output_path, groups):
    lines = [
        "# Stitched Train + Mix Index",
        "",
        "The dashed vertical line in plots is the planned mix boundary at 20M env steps.",
        "",
        "Class labels:",
        "- `fixed_single`: ordinary single v0-style 7x7 run.",
        "- `pseudo_mix_fixed_size`: `scenario_pool` exists, but `dim_room`/`num_boxes` are also forced, so this is not a true size/difficulty mix.",
        "- `padded_single_top_left`: 13x13 top-left padded single-task pretrain.",
        "- `true_mix_padded_top_left`: top-left padded v0/v1/v2 mix.",
        "",
    ]
    for group_name, index in groups.items():
        lines.append(f"## {group_name}")
        for family_name, entries in index.items():
            lines.append(f"### {family_name}")
            for entry in entries:
                if entry["status"] == "missing":
                    lines.append(f"- {entry['prefix']}: missing, offset {entry['offset']}")
                    continue
                cfg = entry["class"]
                lines.append(
                    f"- {entry['prefix']}: {entry['status']}, offset {entry['offset']}, "
                    f"local step {entry['logged_step']}/{entry['target']}, "
                    f"class `{cfg.get('label')}`, pool `{cfg.get('scenario_pool')}`, "
                    f"dim `{cfg.get('dim_room')}`, boxes `{cfg.get('num_boxes')}`, "
                    f"pad `{cfg.get('use_mixed_obs_padding')}`, obs_dim `{cfg.get('obs_dim_room')}`, "
                    f"padding `{cfg.get('mixed_obs_padding_mode')}`, run `{entry['run_dir']}`"
                )
            lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/total-train-plus-mix"))
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_series, old_index = collect(args.results_root, OLD_7X7)
    latest13_series, latest13_index = collect(args.results_root, LATEST_13_TOP_LEFT)
    random13_series, random13_index = collect(args.results_root, RANDOM_13)

    old_path = args.output_dir / "old_7x7_train_plus_pseudo_mix.png"
    latest13_path = args.output_dir / "latest_13x13_top_left_train_plus_mix.png"
    random13_path = args.output_dir / "random_13x13_train_plus_mix.png"
    generated = []
    if plot_series(
        old_series,
        old_path,
        "Original 7x7: pretrain + pseudo-mix",
        args.smoothing,
    ):
        generated.append(old_path)
    if plot_series(
        latest13_series,
        latest13_path,
        "Latest 13x13 top-left padding: pretrain + true mix",
        args.smoothing,
    ):
        generated.append(latest13_path)
    if plot_series(
        random13_series,
        random13_path,
        "13x13 random padding: pretrain + true mix",
        args.smoothing,
    ):
        generated.append(random13_path)

    write_index(
        args.output_dir / "run_index.md",
        {
            "original 7x7 train + pseudo-mix": old_index,
            "latest 13x13 top-left train + mix": latest13_index,
            "13x13 random padding train + mix": random13_index,
        },
    )

    print(f"Saved index: {args.output_dir / 'run_index.md'}")
    for path in generated:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
