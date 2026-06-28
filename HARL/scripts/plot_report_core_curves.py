#!/usr/bin/env python
"""Create compact 2x2 report figures for long Sokoban training curves."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_sokoban_metrics import smooth
from plot_total_train_mix import LATEST_13_TOP_LEFT, OLD_7X7, RANDOM_13, collect


CORE_METRICS = [
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/eval_final_boxes_on_target", "eval boxes on target"),
    ("eval_average_episode_rewards", "eval episode reward"),
]


GROUPS = {
    "01_old_7x7_pseudo_mix_core.png": (
        "Fig. 1 Original 7x7: pretrain + pseudo-mix",
        OLD_7X7,
    ),
    "02_random_13x13_true_mix_core.png": (
        "Fig. 2 13x13 random padding: pretrain + true mix",
        RANDOM_13,
    ),
    "03_top_left_13x13_true_mix_core.png": (
        "Fig. 3 13x13 top-left padding: pretrain + true mix",
        LATEST_13_TOP_LEFT,
    ),
}


def plot_core(series, output_path, title, smoothing, mix_boundary=20_000_000):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, (metric_key, metric_title) in zip(axes.flat, CORE_METRICS):
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
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/report-core-curves"))
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for filename, (title, families) in GROUPS.items():
        series, _ = collect(args.results_root, families)
        output_path = args.output_dir / filename
        plot_core(series, output_path, title, args.smoothing)
        generated.append(output_path)

    index_lines = [
        "# Report Core Curves",
        "",
        "Each figure uses four core metrics in a 2x2 layout. The dashed vertical line marks the planned 20M pretrain/mix boundary.",
        "",
    ]
    for i, path in enumerate(generated, start=1):
        index_lines.append(f"{i}. `{path}`")
    (args.output_dir / "run_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for path in generated:
        print(f"Saved plot: {path}")
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
