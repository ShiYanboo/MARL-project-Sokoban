import argparse
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
    ("sokoban/train_final_boxes_on_target", "train boxes on target"),
    ("eval_average_episode_rewards", "eval episode reward"),
    ("train_episode_rewards/aver_rewards", "train episode reward"),
]


FAMILIES = {
    "v1 nodl-nopush": [
        ("happo-shaped-nodl-nopush", "happo_turn_based_shaped_nodl_nopush", "", 0),
        (
            "happo-shaped-nodl-nopush-resume5m",
            "happo_turn_based_shaped_nodl_nopush_resume5m",
            "",
            5_000_000,
        ),
        (
            "happo-shaped-nodl-nopush-resume10m",
            "happo_turn_based_shaped_nodl_nopush_resume10m",
            "",
            10_000_000,
        ),
        (
            "happo-shaped-nodl-nopush-resume15m",
            "happo_turn_based_shaped_nodl_nopush_resume15m",
            "",
            15_000_000,
        ),
    ],
    "v2 cap12": [
        (
            "happo-shaped-v2",
            "happo_turn_based_shaped_v2_conservative_cap12",
            "-lr1e-4-vlr3e-4-",
            0,
        ),
        (
            "happo-shaped-v2-cap12-resume10m",
            "happo_turn_based_shaped_v2_conservative_cap12_resume10m",
            "",
            5_000_000,
        ),
        (
            "happo-shaped-v2-cap12-resume15m",
            "happo_turn_based_shaped_v2_conservative_cap12_resume15m",
            "",
            10_000_000,
        ),
        (
            "happo-shaped-v2-cap12-resume20m",
            "happo_turn_based_shaped_v2_conservative_cap12_resume20m",
            "",
            15_000_000,
        ),
    ],
    "rnn nodl-nopush": [
        (
            "happo-rnn-bfs-nodl-nopush",
            "happo_turn_based_bfs_nodl_nopush_rnn",
            "",
            0,
        ),
        (
            "happo-rnn-bfs-nodl-nopush-resume10m",
            "happo_turn_based_bfs_nodl_nopush_rnn_resume10m",
            "",
            5_000_000,
        ),
        (
            "happo-rnn-bfs-nodl-nopush-resume15m",
            "happo_turn_based_bfs_nodl_nopush_rnn_resume15m",
            "",
            10_000_000,
        ),
        (
            "happo-rnn-bfs-nodl-nopush-resume20m",
            "happo_turn_based_bfs_nodl_nopush_rnn_resume20m",
            "",
            15_000_000,
        ),
    ],
}


def latest_run_dir(results_root, result_prefix, exp_name, name_filter):
    prefix_dir = results_root / result_prefix
    candidates = []
    for progress_path in prefix_dir.rglob("progress.txt"):
        run_dir = progress_path.parent
        run_path = str(run_dir)
        if exp_name not in run_path:
            continue
        if name_filter and name_filter not in run_path:
            continue
        candidates.append(run_dir)
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
    try:
        metrics.update(load_event_files(run_dir / "logs"))
    except (ImportError, FileNotFoundError, RuntimeError):
        pass
    if "train_episode_rewards/aver_rewards" not in metrics:
        progress_points = load_progress_reward(run_dir)
        if progress_points:
            metrics["train_episode_rewards/aver_rewards"] = progress_points
    return metrics


def stitch_family(results_root, segments, metric_key):
    points = []
    used_runs = []
    for result_prefix, exp_name, name_filter, offset in segments:
        run_dir = latest_run_dir(results_root, result_prefix, exp_name, name_filter)
        if run_dir is None:
            continue
        metric_points = load_run_metrics(run_dir).get(metric_key, [])
        if not metric_points:
            continue
        points.extend((step + offset, value) for step, value in metric_points)
        used_runs.append((result_prefix, run_dir))
    return sorted(points), used_runs


def collect_series(results_root):
    series = {}
    used_runs = {}
    for family_name, segments in FAMILIES.items():
        family_series = {}
        family_used = []
        for metric_key, _ in METRICS:
            points, metric_used = stitch_family(results_root, segments, metric_key)
            family_series[metric_key] = points
            for item in metric_used:
                if item not in family_used:
                    family_used.append(item)
        series[family_name] = family_series
        used_runs[family_name] = family_used
    return series, used_runs


def plot_combined(series, output_path, smoothing):
    available = [
        (metric_key, title)
        for metric_key, title in METRICS
        if any(family_series.get(metric_key) for family_series in series.values())
    ]
    if not available:
        raise RuntimeError("No matching metrics were found.")

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_key, title) in zip(axes.flat, available):
        for family_name, family_series in series.items():
            points = family_series.get(metric_key, [])
            if not points:
                continue
            steps = [step for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=family_name)
        axis.set_title(title)
        axis.set_xlabel("env steps stitched across resumes")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes.flat[len(available):]:
        axis.axis("off")

    fig.suptitle("best family comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir, used_runs):
    lines = ["# Best Family Comparison Plot Index", ""]
    for family_name, runs in used_runs.items():
        lines.append(f"## {family_name}")
        if not runs:
            lines.append("- no runs found")
        for result_prefix, run_dir in runs:
            lines.append(f"- {result_prefix}: `{run_dir}`")
        lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare v1, v2 cap12, and RNN families.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/best-family-comparison-events"),
    )
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series, used_runs = collect_series(args.results_root)
    plot_combined(series, output_dir / "best_family_comparison.png", args.smoothing)
    write_index(output_dir, used_runs)
    print(f"Saved plot to: {output_dir / 'best_family_comparison.png'}")
    print(f"Index: {output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
