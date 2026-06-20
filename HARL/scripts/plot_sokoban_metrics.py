import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GROUPS = {
    "reward_curves.png": [
        ("train_episode_rewards/aver_rewards", "train episode reward"),
        ("eval_average_episode_rewards", "eval episode reward"),
        ("eval_max_episode_rewards", "eval max episode reward"),
        ("critic/average_step_rewards", "average step reward"),
        ("eval_average_episode_length", "eval episode length"),
    ],
    "optimization_curves.png": [
        ("critic/value_loss", "critic value loss"),
        ("critic/critic_grad_norm", "critic grad norm"),
        ("agent0/policy_loss", "agent0 policy loss"),
        ("agent1/policy_loss", "agent1 policy loss"),
        ("agent0/dist_entropy", "agent0 entropy"),
        ("agent1/dist_entropy", "agent1 entropy"),
        ("agent0/ratio", "agent0 ratio"),
        ("agent1/ratio", "agent1 ratio"),
        ("agent0/actor_grad_norm", "agent0 grad norm"),
        ("agent1/actor_grad_norm", "agent1 grad norm"),
    ],
    "sokoban_curves.png": [
        ("sokoban/train_success_rate", "train success rate"),
        ("sokoban/eval_success_rate", "eval success rate"),
        ("sokoban/train_box_completion_ratio", "train box completion ratio"),
        ("sokoban/eval_box_completion_ratio", "eval box completion ratio"),
        ("sokoban/train_box_pushes", "train box pushes"),
        ("sokoban/eval_box_pushes", "eval box pushes"),
        ("sokoban/train_conflict_rate", "train conflict rate"),
        ("sokoban/eval_conflict_rate", "eval conflict rate"),
        ("sokoban/train_noop_rate", "train noop rate"),
        ("sokoban/eval_noop_rate", "eval noop rate"),
    ],
}


def normalize_metric_key(raw_key):
    key = raw_key.replace("\\", "/")
    if "/logs/" in key:
        key = key.split("/logs/", 1)[1]

    parts = [part for part in key.split("/") if part]
    half = len(parts) // 2
    if len(parts) % 2 == 0 and parts[:half] == parts[half:]:
        parts = parts[:half]

    if len(parts) >= 2 and parts[-1] == parts[-2]:
        parts = parts[:-1]

    return "/".join(parts)


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    summary = {}
    for raw_key, entries in raw_data.items():
        key = normalize_metric_key(raw_key)
        points = [(int(step), float(value)) for _, step, value in entries]
        if points:
            summary[key] = points
    return summary


def smooth(values, window):
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_group(summary, output_path, metrics, smoothing):
    available = [(metric_key, title) for metric_key, title in metrics if metric_key in summary]
    if not available:
        return False

    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for axis, (metric_key, title) in zip(axes.flat, available):
        points = summary[metric_key]
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        smoothed = smooth(values, smoothing)

        axis.plot(steps, values, alpha=0.25, linewidth=1.2, label="raw")
        axis.plot(steps, smoothed, linewidth=2.0, label="smoothed")
        axis.set_title(title)
        axis.set_xlabel("env steps")
        axis.grid(alpha=0.3)
        axis.legend()

    for axis in axes.flat[len(available):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_metric_index(summary, output_path):
    lines = ["# Available metrics", ""]
    for metric_name in sorted(summary):
        points = summary[metric_name]
        lines.append(
            f"- {metric_name}: {len(points)} points, step range {points[0][0]} -> {points[-1][0]}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot HARL Sokoban training metrics.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing logs/summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated plots. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=3,
        help="Moving-average window for the smoothed line.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "logs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary file: {summary_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    write_metric_index(summary, output_dir / "metric_index.md")

    generated = []
    for filename, metrics in DEFAULT_GROUPS.items():
        if plot_group(summary, output_dir / filename, metrics, args.smoothing):
            generated.append(filename)

    if not generated:
        raise RuntimeError("No matching metrics were found in summary.json.")

    print(f"Saved plots to: {output_dir}")
    for filename in generated:
        print(f"  - {output_dir / filename}")


if __name__ == "__main__":
    main()
