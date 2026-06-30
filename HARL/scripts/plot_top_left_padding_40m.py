#!/usr/bin/env python
"""Plot 13x13 top-left padding runs through the current pseudo-mix progress."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_report_ablation_figures import (
    M,
    METRIC_TITLES,
    chain_series,
    seg,
)
from plot_sokoban_metrics import smooth


CHAINS = {
    "v1 nodl-nopush": [
        seg("happo13tl-v1-nodl-nopush", 0),
        seg("happo13tl-v1-nodl-nopush-resume10m", 5 * M),
        seg("happo13tl-v1-nodl-nopush-resume15m", 10 * M),
        seg("happo13tl-v1-nodl-nopush-resume20m", 15 * M),
        seg("happo13tl-v1-nodl-nopush-mixv012-r1", 20 * M),
        seg("happo13tl-v1-nodl-nopush-mixv012-r2", 25 * M),
        seg("happo13tl-v1-nodl-nopush-mixv012-r3", 30 * M),
        seg("happo13tl-v1-nodl-nopush-mixv012-r4", 35 * M),
    ],
    "cnn nodl-nopush": [
        seg("happo13tl-cnn-nodl-nopush", 0),
        seg("happo13tl-cnn-nodl-nopush-resume10m", 5 * M),
        seg("happo13tl-cnn-nodl-nopush-resume15m", 10 * M),
        seg("happo13tl-cnn-nodl-nopush-resume20m", 15 * M),
        seg("happo13tl-cnn-nodl-nopush-mixv012-r1", 20 * M),
        seg("happo13tl-cnn-nodl-nopush-mixv012-r2", 25 * M),
        seg("happo13tl-cnn-nodl-nopush-mixv012-r3", 30 * M),
        seg("happo13tl-cnn-nodl-nopush-mixv012-r4", 35 * M),
    ],
    "gru0 nodl-nopush": [
        seg("happo13tl-gru-noaction", 0),
        seg("happo13tl-gru-noaction-resume10m", 5 * M),
        seg("happo13tl-gru-noaction-resume15m", 10 * M),
        seg("happo13tl-gru-noaction-resume20m", 15 * M),
        seg("happo13tl-gru-noaction-mixv012-r1", 20 * M),
        seg("happo13tl-gru-noaction-mixv012-r2", 25 * M),
        seg("happo13tl-gru-noaction-mixv012-r3", 30 * M),
        seg("happo13tl-gru-noaction-mixv012-r4", 35 * M),
    ],
}

METRICS = [
    "sokoban/train_success_rate",
    "sokoban/train_box_completion_ratio",
    "sokoban/eval_success_rate",
    "sokoban/eval_box_completion_ratio",
]


def collect(results_root):
    all_points = {}
    all_entries = {}
    for label, chain in CHAINS.items():
        points_by_metric, entries = chain_series(results_root, chain, METRICS)
        all_points[label] = points_by_metric
        all_entries[label] = entries
    return all_points, all_entries


def write_index(output_path, entries_by_label, points_by_label):
    lines = ["# 13x13 Top-Left Padding 40M Planned Chain", ""]
    lines.append("This figure is not part of the report-figure directory.")
    lines.append("It stitches 5M segments and plots whatever has currently been logged.")
    lines.append("")
    for label, entries in entries_by_label.items():
        max_step = 0
        for metric_points in points_by_label[label].values():
            if metric_points:
                max_step = max(max_step, max(step for step, _ in metric_points))
        lines.append(f"## {label}")
        lines.append(f"- plotted through: {max_step / M:.2f}M stitched steps")
        for entry in entries:
            run_dir = entry.get("run_dir") or "-"
            status = entry.get("status", "missing")
            logged = entry.get("logged_step", "-")
            target = entry.get("target", "-")
            lines.append(
                f"- offset {entry['offset'] / M:.0f}M `{entry['prefix']}`: "
                f"{status}, logged={logged}, target={target}, run={run_dir}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot(output_path, points_by_label, smoothing):
    max_step = 0
    for metrics in points_by_label.values():
        for points in metrics.values():
            if points:
                max_step = max(max_step, max(step for step, _ in points))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
    for axis, metric in zip(axes.flat, METRICS):
        plotted = False
        for label, metrics in points_by_label.items():
            points = metrics.get(metric, [])
            if not points:
                continue
            steps = [step / M for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=label)
            plotted = True
        axis.axvline(20, color="black", alpha=0.25, linestyle="--", linewidth=1.2)
        if max_step > 0:
            axis.set_xlim(0, max_step / M)
        axis.set_title(METRIC_TITLES.get(metric, metric))
        axis.set_xlabel("stitched env steps (M)")
        axis.grid(alpha=0.3)
        if plotted:
            axis.legend(fontsize=8)
    fig.suptitle(
        "13x13 top-left padding: v1 vs CNN vs GRU0 (40M planned, current progress)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/top-left-padding-40m-analysis"),
    )
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    points_by_label, entries_by_label = collect(args.results_root)
    plot(args.output_dir / "01_top_left_v1_cnn_gru0_progress.png", points_by_label, args.smoothing)
    write_index(args.output_dir / "run_index.md", entries_by_label, points_by_label)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
