#!/usr/bin/env python
"""Plot report-focused train/eval key metrics for long Sokoban runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_sokoban_metrics import smooth
from plot_total_train_mix import (
    LATEST_13_TOP_LEFT,
    OLD_7X7,
    RANDOM_13,
    latest_run_dir,
    load_metrics,
)


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


GROUPS = {
    "01_old_7x7_key_metrics.png": (
        "Fig. 1 Original 7x7: key train/eval metrics",
        OLD_7X7,
    ),
    "02_random_13x13_key_metrics.png": (
        "Fig. 2 13x13 random padding: key train/eval metrics",
        RANDOM_13,
    ),
    "03_top_left_13x13_key_metrics.png": (
        "Fig. 3 13x13 top-left padding: key train/eval metrics",
        LATEST_13_TOP_LEFT,
    ),
}


def collect_key_metrics(results_root, families):
    all_series = {}
    used = {}
    for family_name, segments in families.items():
        series = {metric_key: [] for metric_key, _ in KEY_METRICS}
        family_used = []
        for prefix, offset in segments:
            run_dir = latest_run_dir(results_root, prefix)
            family_used.append((prefix, run_dir))
            if run_dir is None:
                continue
            run_metrics = load_metrics(run_dir)
            for metric_key, _ in KEY_METRICS:
                points = run_metrics.get(metric_key, [])
                series[metric_key].extend((step + offset, value) for step, value in points)
        all_series[family_name] = {
            metric_key: sorted(points) for metric_key, points in series.items()
        }
        used[family_name] = family_used
    return all_series, used


def plot_key_metrics(series, output_path, title, smoothing, mix_boundary=20_000_000):
    fig, axes = plt.subplots(4, 2, figsize=(16, 17))
    for axis, (metric_key, metric_title) in zip(axes.flat, KEY_METRICS):
        for family_name, family_series in series.items():
            points = family_series.get(metric_key, [])
            if not points:
                continue
            steps = [step / 1_000_000 for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=family_name)
        axis.axvline(mix_boundary / 1_000_000, color="black", alpha=0.25, linestyle="--")
        axis.set_title(metric_title)
        axis.set_xlabel("stitched env steps (M)")
        axis.grid(alpha=0.3)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir, used_groups):
    lines = [
        "# Report Key Metrics",
        "",
        "Metrics: train/eval success rate, train/eval box completion ratio, train/eval base reward per step, critic value loss, and eval episode reward.",
        "The dashed vertical line marks the planned 20M pretrain/mix boundary.",
        "",
    ]
    for group_name, used in used_groups.items():
        lines.append(f"## {group_name}")
        for family_name, runs in used.items():
            lines.append(f"### {family_name}")
            for prefix, run_dir in runs:
                if run_dir is None:
                    lines.append(f"- {prefix}: missing")
                else:
                    lines.append(f"- {prefix}: `{run_dir}`")
            lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/report-key-metrics"))
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_groups = {}
    for filename, (title, families) in GROUPS.items():
        series, used = collect_key_metrics(args.results_root, families)
        output_path = args.output_dir / filename
        plot_key_metrics(series, output_path, title, args.smoothing)
        used_groups[title] = used
        print(f"Saved plot: {output_path}")
    write_index(args.output_dir, used_groups)
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
