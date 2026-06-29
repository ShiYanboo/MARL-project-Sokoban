#!/usr/bin/env python
"""Plot key metrics for run17 HAHyPO experiments."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sokoban_metrics import load_event_files, load_summary, smooth


KEY_METRICS = [
    ("sokoban/train_success_rate", "train success rate"),
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/train_box_completion_ratio", "train box completion ratio"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/train_mean_base_reward", "train base reward / step"),
    ("sokoban/eval_mean_base_reward", "eval base reward / step"),
    ("critic/value_loss", "critic value loss"),
    ("eval_average_episode_rewards", "eval episode reward"),
]


VARIANTS = {
    "baseline": "baseline",
    "v1-nodl-nopush": "v1_nodl_nopush",
    "gru-hist8": "gru_hist8",
}


ALPHAS = [
    ("alpha=0.2", "a02"),
    ("alpha=0.5", "a05"),
    ("alpha=1.0", "a10"),
]


SEGMENTS = [
    ("5m", "", 0),
    ("resume10m", "-resume10m", 5_000_000),
    ("resume15m", "-resume15m", 10_000_000),
    ("resume20m", "-resume20m", 15_000_000),
]


def latest_run_dir(results_root, prefix):
    prefix_dir = results_root / prefix
    if not prefix_dir.exists():
        return None
    candidates = [path.parent for path in prefix_dir.rglob("progress.txt")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_progress_reward(run_dir):
    progress_path = run_dir / "progress.txt"
    if not progress_path.exists():
        return []
    points = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        try:
            points.append((int(float(parts[0])), float(parts[1])))
        except ValueError:
            continue
    return points


def load_run_metrics(run_dir):
    metrics = {}
    summary_path = run_dir / "logs" / "summary.json"
    if summary_path.exists():
        metrics.update(load_summary(summary_path))
    log_dir = run_dir / "logs"
    if log_dir.exists():
        try:
            event_metrics = load_event_files(log_dir)
        except (ImportError, FileNotFoundError, RuntimeError):
            event_metrics = {}
        for key, points in event_metrics.items():
            metrics.setdefault(key, points)
    if "train_episode_rewards/aver_rewards" not in metrics:
        progress_points = load_progress_reward(run_dir)
        if progress_points:
            metrics["train_episode_rewards/aver_rewards"] = progress_points
    return metrics


def collect_series(results_root):
    series = {}
    used_runs = {}
    for variant_slug in VARIANTS:
        variant_series = {}
        variant_used = {}
        for alpha_label, alpha_slug in ALPHAS:
            family_prefix = f"hahypo-{alpha_slug}-{variant_slug}"
            metric_series = {metric_key: [] for metric_key, _ in KEY_METRICS}
            used = []
            for segment_label, suffix, offset in SEGMENTS:
                run_dir = latest_run_dir(results_root, f"{family_prefix}{suffix}")
                used.append((segment_label, f"{family_prefix}{suffix}", run_dir))
                if run_dir is None:
                    continue
                run_metrics = load_run_metrics(run_dir)
                for metric_key, _ in KEY_METRICS:
                    points = run_metrics.get(metric_key, [])
                    metric_series[metric_key].extend(
                        (step + offset, value) for step, value in points
                    )
            variant_series[alpha_label] = {
                metric_key: sorted(points)
                for metric_key, points in metric_series.items()
            }
            variant_used[alpha_label] = used
        series[variant_slug] = variant_series
        used_runs[variant_slug] = variant_used
    return series, used_runs


def plot_variant(variant_slug, variant_series, output_path, smoothing):
    available = [
        (metric_key, title)
        for metric_key, title in KEY_METRICS
        if any(alpha_series.get(metric_key) for alpha_series in variant_series.values())
    ]
    if not available:
        return False

    rows = int(np.ceil(len(available) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, 2)

    for axis, (metric_key, title) in zip(axes.flat, available):
        for alpha_label, alpha_series in variant_series.items():
            points = alpha_series.get(metric_key, [])
            if not points:
                continue
            steps = [step / 1_000_000 for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=alpha_label)
        axis.set_title(title)
        axis.set_xlabel("stitched env steps (M)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    for axis in axes.flat[len(available):]:
        axis.axis("off")

    fig.suptitle(f"run17 HAHyPO: {variant_slug}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_all(series, output_path, smoothing):
    rows = len(KEY_METRICS)
    fig, axes = plt.subplots(rows, 1, figsize=(15, 3.2 * rows), sharex=False)
    axes = np.atleast_1d(axes)
    for axis, (metric_key, title) in zip(axes, KEY_METRICS):
        for variant_slug, variant_series in series.items():
            for alpha_label, alpha_series in variant_series.items():
                points = alpha_series.get(metric_key, [])
                if not points:
                    continue
                steps = [step / 1_000_000 for step, _ in points]
                values = [value for _, value in points]
                axis.plot(
                    steps,
                    smooth(values, smoothing),
                    linewidth=1.5,
                    label=f"{variant_slug} {alpha_label}",
                )
        axis.set_title(title)
        axis.set_xlabel("stitched env steps (M)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6, ncol=3)
    fig.suptitle("run17 HAHyPO: all variants", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir, used_runs):
    lines = [
        "# run17 HAHyPO Plot Index",
        "",
        "Each curve stitches available 5M segments in order. Missing later segments are left absent.",
        "",
    ]
    for variant_slug, alpha_runs in used_runs.items():
        lines.append(f"## {variant_slug}")
        for alpha_label, runs in alpha_runs.items():
            lines.append(f"### {alpha_label}")
            for segment_label, prefix, run_dir in runs:
                if run_dir is None:
                    lines.append(f"- {segment_label} `{prefix}`: missing")
                else:
                    lines.append(f"- {segment_label} `{prefix}`: `{run_dir}`")
            lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot run17 HAHyPO key metrics.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/run17-hahypo"))
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    series, used_runs = collect_series(args.results_root)
    for variant_slug, filename in VARIANTS.items():
        output_path = args.output_dir / f"{filename}.png"
        if plot_variant(variant_slug, series[variant_slug], output_path, args.smoothing):
            print(f"Saved plot: {output_path}")
    plot_all(series, args.output_dir / "all_run17_hahypo.png", args.smoothing)
    print(f"Saved plot: {args.output_dir / 'all_run17_hahypo.png'}")
    write_index(args.output_dir, used_runs)
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
