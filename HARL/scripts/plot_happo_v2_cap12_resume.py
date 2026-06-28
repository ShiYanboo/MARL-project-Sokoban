import argparse
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


SEGMENTS = [
    (
        "base",
        "happo-shaped-v2",
        "happo_turn_based_shaped_v2_conservative_cap12",
        "-lr1e-4-vlr3e-4-",
        0,
    ),
    (
        "resume10m",
        "happo-shaped-v2-cap12-resume10m",
        "happo_turn_based_shaped_v2_conservative_cap12_resume10m",
        "",
        5_000_000,
    ),
    (
        "resume15m",
        "happo-shaped-v2-cap12-resume15m",
        "happo_turn_based_shaped_v2_conservative_cap12_resume15m",
        "",
        10_000_000,
    ),
    (
        "resume20m",
        "happo-shaped-v2-cap12-resume20m",
        "happo_turn_based_shaped_v2_conservative_cap12_resume20m",
        "",
        15_000_000,
    ),
]


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


def stitch_metric(results_root, metric_key):
    points = []
    used_runs = []
    for label, prefix, exp_name, name_filter, offset in SEGMENTS:
        run_dir = latest_run_dir(results_root, prefix, exp_name, name_filter)
        if run_dir is None:
            continue
        metrics = load_run_metrics(run_dir)
        metric_points = metrics.get(metric_key, [])
        if not metric_points:
            continue
        points.extend((step + offset, value) for step, value in metric_points)
        used_runs.append((label, run_dir))
    return sorted(points), used_runs


def plot_metrics(series_by_metric, output_path, smoothing):
    available = [(key, title) for key, title in METRICS if series_by_metric.get(key)]
    if not available:
        raise RuntimeError("No matching metrics were found.")

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_key, title) in zip(axes.flat, available):
        points = series_by_metric[metric_key]
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        axis.plot(steps, values, alpha=0.25, linewidth=1.0, label="raw")
        axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label="smoothed")
        axis.set_title(title)
        axis.set_xlabel("env steps stitched across cap12 resumes")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes.flat[len(available):]:
        axis.axis("off")

    fig.suptitle("happo-shaped-v2 cap12 resume")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_index(output_dir, used_runs_by_metric):
    lines = ["# Happo V2 Cap12 Resume Plot Index", ""]
    seen = []
    for metric_key, used_runs in used_runs_by_metric.items():
        lines.append(f"## {metric_key}")
        for label, run_dir in used_runs:
            item = (label, run_dir)
            if item not in seen:
                seen.append(item)
            lines.append(f"- {label}: `{run_dir}`")
        lines.append("")

    lines.append("## Unique runs")
    for label, run_dir in seen:
        lines.append(f"- {label}: `{run_dir}`")
    lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot stitched v2 cap12 resume curves.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/happo-shaped-v2-cap12-resume-events"),
    )
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series_by_metric = {}
    used_runs_by_metric = {}
    for metric_key, _ in METRICS:
        points, used_runs = stitch_metric(args.results_root, metric_key)
        series_by_metric[metric_key] = points
        used_runs_by_metric[metric_key] = used_runs

    plot_metrics(series_by_metric, output_dir / "cap12_resume.png", args.smoothing)
    write_index(output_dir, used_runs_by_metric)
    print(f"Saved plot to: {output_dir / 'cap12_resume.png'}")
    print(f"Index: {output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
