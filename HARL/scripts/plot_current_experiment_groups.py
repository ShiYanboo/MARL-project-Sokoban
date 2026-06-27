#!/usr/bin/env python
"""Plot current Sokoban experiment groups from event files."""

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


COMPLETED_20M_FAMILIES = {
    "v1 strongpot": [
        ("happo-shaped-strongpot", 0),
        ("happo-shaped-strongpot-resume5m", 5_000_000),
        ("happo-shaped-strongpot-resume10m", 10_000_000),
        ("happo-shaped-strongpot-resume15m", 15_000_000),
    ],
    "v1 nodeadlock": [
        ("happo-shaped-nodeadlock", 0),
        ("happo-shaped-nodeadlock-resume5m", 5_000_000),
        ("happo-shaped-nodeadlock-resume10m", 10_000_000),
        ("happo-shaped-nodeadlock-resume15m", 15_000_000),
    ],
    "v1 nodl-nopush": [
        ("happo-shaped-nodl-nopush", 0),
        ("happo-shaped-nodl-nopush-resume5m", 5_000_000),
        ("happo-shaped-nodl-nopush-resume10m", 10_000_000),
        ("happo-shaped-nodl-nopush-resume15m", 15_000_000),
    ],
    "v1 noshape": [
        ("happo-shaped-noshape", 0),
        ("happo-shaped-noshape-resume5m", 5_000_000),
        ("happo-shaped-noshape-resume10m", 10_000_000),
        ("happo-shaped-noshape-resume15m", 15_000_000),
    ],
    "v2 cap12": [
        ("happo-shaped-v2", 0),
        ("happo-shaped-v2-cap12-resume10m", 5_000_000),
        ("happo-shaped-v2-cap12-resume15m", 10_000_000),
        ("happo-shaped-v2-cap12-resume20m", 15_000_000),
    ],
    "gru nodl-nopush hist8": [
        ("happo-rnn-bfs-nodl-nopush", 0),
        ("happo-rnn-bfs-nodl-nopush-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-resume20m", 15_000_000),
    ],
}


NEW_5M_ABLATIONS = {
    "deadlock 0.05": "happo-shaped-deadlock005",
    "deadlock 0.10": "happo-shaped-deadlock010",
    "deadlock 0.20": "happo-shaped-deadlock020",
    "deadlock 0.40": "happo-shaped-deadlock040",
    "mlp hist8": "happo-bfs-nodl-nopush-mlp-hist8",
    "gru nohist": "happo-rnn-bfs-nodl-nopush-nohist",
    "gru hist1": "happo-rnn-bfs-nodl-nopush-hist1-thr32",
    "gru hist2": "happo-rnn-bfs-nodl-nopush-hist2",
    "gru hist8": "happo-rnn-bfs-nodl-nopush",
    "gru hist16": "happo-rnn-bfs-nodl-nopush-hist16",
    "gru nohist thr48": "happo-rnn-bfs-nodl-nopush-nohist-thr48",
    "gru hist1 thr48": "happo-rnn-bfs-nodl-nopush-hist1-thr48",
    "gru hist8 bigcap thr48": "happo-rnn-bfs-nodl-nopush-hist8-bigcap-thr48",
    "gru latelr25m": "happo-rnn-bfs-nodl-nopush-latelr-resume25m",
    "v2 cnn": "happo-shaped-v2-cnn",
}


HAPPO13_5M = {
    "13 baseline": "happo13-baseline-original",
    "13 noshape": "happo13-v1-noshape",
    "13 nodl-nopush": "happo13-v1-nodl-nopush",
    "13 gru noact": "happo13-gru-noaction",
    "13 gru 8act": "happo13-gru-8action",
    "13 gru 1act": "happo13-gru-1action",
    "13 cnn": "happo13-cnn-nodl-nopush",
}


MIX_FAMILIES = {
    "v1 strongpot mix": [
        ("happo-shaped-strongpot-mixv012-r1", 0),
        ("happo-shaped-strongpot-mixv012-r2", 5_000_000),
        ("happo-shaped-strongpot-mixv012-r3", 10_000_000),
        ("happo-shaped-strongpot-mixv012-r4", 15_000_000),
    ],
    "v1 nodeadlock mix": [
        ("happo-shaped-nodeadlock-mixv012-r1", 0),
        ("happo-shaped-nodeadlock-mixv012-r2", 5_000_000),
        ("happo-shaped-nodeadlock-mixv012-r3", 10_000_000),
        ("happo-shaped-nodeadlock-mixv012-r4", 15_000_000),
    ],
    "v1 nodl-nopush mix": [
        ("happo-shaped-nodl-nopush-mixv012-r1", 0),
        ("happo-shaped-nodl-nopush-mixv012-r2", 5_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r3", 10_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r4", 15_000_000),
    ],
    "v1 noshape mix": [
        ("happo-shaped-noshape-mixv012-r1", 0),
        ("happo-shaped-noshape-mixv012-r2", 5_000_000),
        ("happo-shaped-noshape-mixv012-r3", 10_000_000),
        ("happo-shaped-noshape-mixv012-r4", 15_000_000),
    ],
    "v2 cap12 mix": [
        ("happo-shaped-v2-cap12-mixv012-r1", 0),
        ("happo-shaped-v2-cap12-mixv012-r2", 5_000_000),
    ],
    "gru hist8 mix": [
        ("happo-rnn-bfs-nodl-nopush-mixv012-r1", 0),
        ("happo-rnn-bfs-nodl-nopush-mixv012-r2", 5_000_000),
    ],
}


def latest_run_dir(results_root, result_prefix):
    prefix_dir = results_root / result_prefix
    candidates = [path.parent for path in prefix_dir.rglob("progress.txt")]
    if not candidates:
        candidates = [path.parent for path in prefix_dir.rglob("config.json")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def max_logged_step(run_dir):
    progress_path = run_dir / "progress.txt"
    max_step = 0
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            patterns = [
                r"total num timesteps\s+(\d+)/(\d+)",
                r"^\s*(\d+(?:\.\d+)?)\s*,",
            ]
            for pattern in patterns:
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


def is_complete(run_dir, threshold=0.95):
    target = target_steps(run_dir)
    if not target:
        return False
    return max_logged_step(run_dir) >= threshold * target


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


def stitch_segments(results_root, segments, metric_key, require_complete=True):
    points = []
    used = []
    for prefix, offset in segments:
        run_dir = latest_run_dir(results_root, prefix)
        if run_dir is None:
            used.append((prefix, None, "missing", 0, None))
            continue
        step = max_logged_step(run_dir)
        target = target_steps(run_dir)
        complete = is_complete(run_dir)
        used.append((prefix, run_dir, "complete" if complete else "partial", step, target))
        if require_complete and not complete:
            continue
        metric_points = load_metrics(run_dir).get(metric_key, [])
        points.extend((step_value + offset, value) for step_value, value in metric_points)
    return sorted(points), used


def collect_stitched(results_root, families, require_complete=True):
    all_series = {}
    used = {}
    for name, segments in families.items():
        series = {}
        family_used = []
        for metric_key, _ in METRICS:
            points, metric_used = stitch_segments(
                results_root, segments, metric_key, require_complete=require_complete
            )
            series[metric_key] = points
            for item in metric_used:
                if item not in family_used:
                    family_used.append(item)
        all_series[name] = series
        used[name] = family_used
    return all_series, used


def collect_single_runs(results_root, runs, require_complete=True):
    families = {name: [(prefix, 0)] for name, prefix in runs.items()}
    return collect_stitched(results_root, families, require_complete=require_complete)


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
        for name, series in all_series.items():
            points = series.get(metric_key, [])
            if not points:
                continue
            steps = [step for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=name)
        axis.set_title(metric_title)
        axis.set_xlabel("env steps")
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


def write_index(output_dir, groups):
    lines = ["# Current Experiment Plot Index", ""]
    for group_name, used in groups.items():
        lines.append(f"## {group_name}")
        for family_name, runs in used.items():
            lines.append(f"### {family_name}")
            for prefix, run_dir, status, step, target in runs:
                target_text = "?" if target is None else str(target)
                lines.append(
                    f"- {prefix}: {status}, step {step}/{target_text}, `{run_dir}`"
                )
            lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/current-experiments"))
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_20m, used_20m = collect_stitched(
        args.results_root, COMPLETED_20M_FAMILIES, require_complete=True
    )
    new_5m, used_new_5m = collect_single_runs(
        args.results_root, NEW_5M_ABLATIONS, require_complete=True
    )
    happo13_5m, used_happo13 = collect_single_runs(
        args.results_root, HAPPO13_5M, require_complete=False
    )
    mix_completed, used_mix = collect_stitched(
        args.results_root, MIX_FAMILIES, require_complete=True
    )

    generated = []
    if plot_series(
        completed_20m,
        output_dir / "non13_completed_20m_premix.png",
        "non-13x13 completed pre-mix 20M families",
        args.smoothing,
    ):
        generated.append(output_dir / "non13_completed_20m_premix.png")
    if plot_series(
        new_5m,
        output_dir / "non13_new_5m_ablations.png",
        "non-13x13 new completed 5M ablations",
        args.smoothing,
    ):
        generated.append(output_dir / "non13_new_5m_ablations.png")
    if plot_series(
        happo13_5m,
        output_dir / "happo13_current_5m_partial.png",
        "13x13 random-padding current 5M runs",
        args.smoothing,
    ):
        generated.append(output_dir / "happo13_current_5m_partial.png")
    if plot_series(
        mix_completed,
        output_dir / "mix_completed_rounds_only.png",
        "completed mix012 rounds only",
        args.smoothing,
    ):
        generated.append(output_dir / "mix_completed_rounds_only.png")

    write_index(
        output_dir,
        {
            "non13 completed 20m": used_20m,
            "non13 new 5m ablations": used_new_5m,
            "happo13 current 5m partial": used_happo13,
            "mix completed rounds only": used_mix,
        },
    )
    print(f"Saved plots to: {output_dir}")
    for path in generated:
        print(f"  - {path}")
    print(f"Index: {output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
