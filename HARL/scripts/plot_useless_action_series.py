#!/usr/bin/env python
"""Plot exploratory useless-action-penalty curves."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_report_ablation_figures import CHAINS, M, chain_series
from plot_sokoban_metrics import smooth


def seg(prefix, offset):
    return {"prefix": prefix, "offset": offset, "contains": None}


def long_chain(prefix):
    return [
        seg(prefix, 0),
        seg(f"{prefix}-resume10m", 5 * M),
        seg(f"{prefix}-resume15m", 10 * M),
        seg(f"{prefix}-resume20m", 15 * M),
        seg(f"{prefix}-mixv012-r1", 20 * M),
        seg(f"{prefix}-mixv012-r2", 25 * M),
        seg(f"{prefix}-mixv012-r3", 30 * M),
        seg(f"{prefix}-mixv012-r4", 35 * M),
    ]


CHAINS = {
    **CHAINS,
    "v1-useless-0.1": long_chain("happo-useless-p01-v1-nodl-nopush"),
    "v1-useless-0.2": long_chain("happo-useless-p02-v1-nodl-nopush"),
    "gru0-useless-0.1": long_chain("happo-useless-p01-gru0-nodl-nopush"),
    "gru0-useless-0.2": long_chain("happo-useless-p02-gru0-nodl-nopush"),
    "v1-deadlock0.2-useless0.2": long_chain("happo-useless-p02-v1-nopush-dl02"),
}


METRICS = [
    ("sokoban/train_success_rate", "train success rate"),
    ("sokoban/train_box_completion_ratio", "train box completion ratio"),
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/train_useless_action_rate", "train useless action rate"),
    ("critic/value_loss", "critic value loss"),
]


TOTAL_FAMILIES = {
    "v1 useless 0.1": "v1-useless-0.1",
    "v1 useless 0.2": "v1-useless-0.2",
    "gru0 useless 0.1": "gru0-useless-0.1",
    "gru0 useless 0.2": "gru0-useless-0.2",
    "v1 dl0.2 + useless0.2": "v1-deadlock0.2-useless0.2",
}


V1_COMPARISON = {
    "v1 nodl-nopush": "v1-nodl-nopush",
    "v1 useless 0.1": "v1-useless-0.1",
    "v1 useless 0.2": "v1-useless-0.2",
    "v1 dl0.2 + useless0.2": "v1-deadlock0.2-useless0.2",
}


GRU_COMPARISON = {
    "gru0 baseline": "gru0",
    "gru0 useless 0.1": "gru0-useless-0.1",
    "gru0 useless 0.2": "gru0-useless-0.2",
}


PUSH_MOVE_COMPARISON = {
    "v1 nodl-nopush": "v1-nodl-nopush",
    "gru0 baseline": "gru0",
    "v1 useless 0.1": "v1-useless-0.1",
    "v1 useless 0.2": "v1-useless-0.2",
    "gru0 useless 0.1": "gru0-useless-0.1",
    "gru0 useless 0.2": "gru0-useless-0.2",
}


PUSH_MOVE_METRICS = [
    ("sokoban/train_box_pushes", "train box pushes per finished episode"),
    ("sokoban/train_player_moves", "train player moves per finished episode"),
]


def collect_family(results_root, chain_name, metrics):
    data = {}
    entries = None
    for metric, _ in metrics:
        series, metric_entries = chain_series(results_root, CHAINS[chain_name], [metric])
        data[metric] = series.get(metric, [])
        if entries is None:
            entries = metric_entries
    return data, entries or []


def common_xlim(series_by_label, metrics):
    maxima = []
    for metric, _ in metrics:
        for data in series_by_label.values():
            points = data.get(metric, [])
            if points:
                maxima.append(max(step for step, _ in points))
    return min(maxima) if maxima else None


def plot_grid(series_by_label, metrics, title, output_path, smoothing, trim_to_shortest=False):
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.0 * rows), squeeze=False)
    xlim = common_xlim(series_by_label, metrics) if trim_to_shortest else None
    for axis, (metric, metric_title) in zip(axes.flat, metrics):
        for label, data in series_by_label.items():
            points = data.get(metric, [])
            if xlim is not None:
                points = [(step, value) for step, value in points if step <= xlim]
            if not points:
                continue
            steps = [step / M for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=1.8, label=label)
        if xlim is None or xlim >= 20 * M:
            axis.axvline(20, color="black", alpha=0.25, linestyle="--", linewidth=1.1)
        if xlim is not None:
            axis.set_xlim(0, xlim / M)
        axis.set_title(metric_title)
        axis.set_xlabel("stitched env steps (M)")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8, ncol=2)
    fig.suptitle(title, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir, generated, used):
    lines = [
        "# Useless Action Analysis",
        "",
        "Exploratory plots only; these are not part of the formal report figure set yet.",
        "The dashed vertical line marks the 20M pretrain / pseudo-mix boundary when visible.",
        "",
        "## Figures",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.extend(["", "## Sources", ""])
    for family, entries in used.items():
        lines.append(f"### {family}")
        for entry in entries:
            if entry["status"] == "missing":
                lines.append(f"- `{entry['prefix']}`: missing")
            else:
                lines.append(
                    f"- `{entry['prefix']}`: {entry['status']}, "
                    f"local {entry.get('logged_step')}/{entry.get('target')}, "
                    f"offset {entry['offset']}, run `{entry['run_dir']}`"
                )
        lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/useless-action-analysis"))
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    used = {}

    total_series = {}
    for label, chain_name in TOTAL_FAMILIES.items():
        total_series[label], used[label] = collect_family(args.results_root, chain_name, METRICS)
    total_path = args.output_dir / "01_useless_action_series_overview.png"
    plot_grid(
        total_series,
        METRICS,
        "Useless-action penalty series overview",
        total_path,
        args.smoothing,
        trim_to_shortest=False,
    )
    generated.append(total_path)

    comparison_metrics = METRICS[:4] + [METRICS[5]]
    v1_series = {}
    for label, chain_name in V1_COMPARISON.items():
        v1_series[label], used[label] = collect_family(args.results_root, chain_name, comparison_metrics)
    v1_path = args.output_dir / "02_useless_action_vs_v1_nodl_nopush.png"
    plot_grid(
        v1_series,
        comparison_metrics,
        "Useless-action penalty vs v1 nodl-nopush baseline",
        v1_path,
        args.smoothing,
        trim_to_shortest=False,
    )
    generated.append(v1_path)

    gru_series = {}
    for label, chain_name in GRU_COMPARISON.items():
        gru_series[label], used[label] = collect_family(args.results_root, chain_name, comparison_metrics)
    gru_path = args.output_dir / "03_useless_action_vs_gru0_baseline.png"
    plot_grid(
        gru_series,
        comparison_metrics,
        "Useless-action penalty vs GRU0 baseline",
        gru_path,
        args.smoothing,
        trim_to_shortest=False,
    )
    generated.append(gru_path)

    push_move_series = {}
    for label, chain_name in PUSH_MOVE_COMPARISON.items():
        push_move_series[label], used[label] = collect_family(
            args.results_root, chain_name, PUSH_MOVE_METRICS
        )
    push_move_path = args.output_dir / "04_pushes_and_player_moves.png"
    plot_grid(
        push_move_series,
        PUSH_MOVE_METRICS,
        "Box pushes and player moves",
        push_move_path,
        args.smoothing,
        trim_to_shortest=False,
    )
    generated.append(push_move_path)

    write_index(args.output_dir, generated, used)
    for path in generated:
        print(f"Saved plot: {path}")
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
