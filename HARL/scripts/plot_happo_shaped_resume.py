import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sokoban_metrics import load_event_files, load_summary, smooth


CATEGORIES = {
    "strongpot": [
        ("happo-shaped-strongpot", 0),
        ("happo-shaped-strongpot-resume5m", 5_000_000),
        ("happo-shaped-strongpot-resume10m", 10_000_000),
        ("happo-shaped-strongpot-resume15m", 15_000_000),
    ],
    "nodeadlock": [
        ("happo-shaped-nodeadlock", 0),
        ("happo-shaped-nodeadlock-resume5m", 5_000_000),
        ("happo-shaped-nodeadlock-resume10m", 10_000_000),
        ("happo-shaped-nodeadlock-resume15m", 15_000_000),
    ],
    "nodl-nopush": [
        ("happo-shaped-nodl-nopush", 0),
        ("happo-shaped-nodl-nopush-resume5m", 5_000_000),
        ("happo-shaped-nodl-nopush-resume10m", 10_000_000),
        ("happo-shaped-nodl-nopush-resume15m", 15_000_000),
    ],
    "noshape": [
        ("happo-shaped-noshape", 0),
        ("happo-shaped-noshape-resume5m", 5_000_000),
        ("happo-shaped-noshape-resume10m", 10_000_000),
        ("happo-shaped-noshape-resume15m", 15_000_000),
    ],
}


DEFAULT_METRICS = [
    ("sokoban/train_success_rate", "train success rate"),
    ("sokoban/eval_success_rate", "eval success rate"),
    ("sokoban/train_box_completion_ratio", "train box completion ratio"),
    ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
    ("sokoban/train_mean_base_reward", "train base reward / step"),
    ("sokoban/eval_mean_base_reward", "eval base reward / step"),
    ("critic/value_loss", "critic value loss"),
    ("eval_average_episode_rewards", "eval episode reward"),
]


def latest_run_dir(results_root, result_prefix):
    prefix_dir = results_root / result_prefix
    candidates = [
        path.parent
        for path in prefix_dir.rglob("progress.txt")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_progress_reward(run_dir):
    progress_path = run_dir / "progress.txt"
    if not progress_path.exists():
        return []
    points = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            points.append((int(float(parts[0])), float(parts[1])))
        except ValueError:
            continue
    return points


def load_metric_points(run_dir, metric_key):
    summary_path = run_dir / "logs" / "summary.json"
    metrics = {}
    if summary_path.exists():
        metrics = load_summary(summary_path)
    if metric_key not in metrics:
        try:
            metrics.update(load_event_files(run_dir / "logs"))
        except (ImportError, FileNotFoundError, RuntimeError):
            pass
    if metric_key in metrics:
        return metrics[metric_key]
    if metric_key == "train_episode_rewards/aver_rewards":
        return load_progress_reward(run_dir)
    return []


def stitch_category(results_root, segments, metric_key):
    stitched = []
    used_runs = []
    for result_prefix, offset in segments:
        run_dir = latest_run_dir(results_root, result_prefix)
        if run_dir is None:
            continue
        points = load_metric_points(run_dir, metric_key)
        if not points:
            continue
        stitched.extend((step + offset, value) for step, value in points)
        used_runs.append((result_prefix, run_dir))
    return sorted(stitched), used_runs


def plot_category(category_name, series_by_metric, metrics, output_dir, smoothing):
    available_metrics = [
        (metric_key, title)
        for metric_key, title in metrics
        if series_by_metric.get(metric_key)
    ]
    if not available_metrics:
        return False

    cols = 2
    rows = int(np.ceil(len(available_metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_key, title) in zip(axes.flat, available_metrics):
        points = series_by_metric.get(metric_key, [])
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        axis.plot(steps, values, alpha=0.25, linewidth=1.0, label="raw")
        axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label="smoothed")
        axis.set_title(title)
        axis.set_xlabel("env steps stitched across resumes")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes.flat[len(available_metrics):]:
        axis.axis("off")

    fig.suptitle(category_name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{category_name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_combined(all_series, metrics, output_dir, smoothing):
    available_metrics = [
        (metric_key, title)
        for metric_key, title in metrics
        if any(series_by_metric.get(metric_key) for series_by_metric in all_series.values())
    ]
    if not available_metrics:
        return False

    cols = 2
    rows = int(np.ceil(len(available_metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_key, title) in zip(axes.flat, available_metrics):
        for category_name, series_by_metric in all_series.items():
            points = series_by_metric.get(metric_key, [])
            if not points:
                continue
            steps = [step for step, _ in points]
            values = [value for _, value in points]
            axis.plot(steps, smooth(values, smoothing), linewidth=2.0, label=category_name)
        axis.set_title(title)
        axis.set_xlabel("env steps stitched across resumes")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend()

    for axis in axes.flat[len(available_metrics):]:
        axis.axis("off")

    fig.suptitle("happo-shaped resume comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_categories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_index(output_dir, used_runs_by_category):
    lines = ["# Happo Shaped Resume Plot Index", ""]
    for category_name, used_runs in used_runs_by_category.items():
        lines.append(f"## {category_name}")
        if not used_runs:
            lines.append("- no runs found")
        for result_prefix, run_dir in used_runs:
            lines.append(f"- {result_prefix}: `{run_dir}`")
        lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Plot stitched happo-shaped initial/resume curves without TensorBoard."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/happo-shaped-resume"),
    )
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_series = {}
    used_runs_by_category = {}
    for category_name, segments in CATEGORIES.items():
        series_by_metric = {}
        used_runs = []
        for metric_key, _ in DEFAULT_METRICS:
            points, metric_used_runs = stitch_category(args.results_root, segments, metric_key)
            series_by_metric[metric_key] = points
            for item in metric_used_runs:
                if item not in used_runs:
                    used_runs.append(item)
        all_series[category_name] = series_by_metric
        used_runs_by_category[category_name] = used_runs
        plot_category(category_name, series_by_metric, DEFAULT_METRICS, output_dir, args.smoothing)

    plot_combined(all_series, DEFAULT_METRICS, output_dir, args.smoothing)
    write_index(output_dir, used_runs_by_category)
    print(f"Saved plots to: {output_dir}")
    print(f"Index: {output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
