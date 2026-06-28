#!/usr/bin/env python
"""Compare full v2 cap12 pseudo-mix workflow against key 7x7 baselines."""

import argparse
import json
import re
from pathlib import Path

from plot_report_key_metrics import KEY_METRICS, collect_key_metrics, plot_key_metrics
from plot_total_train_mix import latest_run_dir


FAMILIES = {
    "baseline shared reward": [
        ("happo-baseline-original", 0),
        ("happo-baseline-original-resume10m", 5_000_000),
        ("happo-baseline-original-resume15m", 10_000_000),
        ("happo-baseline-original-resume20m", 15_000_000),
        ("happo-baseline-original-mixv012-r1", 20_000_000),
        ("happo-baseline-original-mixv012-r2", 25_000_000),
        ("happo-baseline-original-mixv012-r3", 30_000_000),
        ("happo-baseline-original-mixv012-r4", 35_000_000),
    ],
    "v1 nodl-nopush": [
        ("happo-shaped-nodl-nopush", 0),
        ("happo-shaped-nodl-nopush-resume5m", 5_000_000),
        ("happo-shaped-nodl-nopush-resume10m", 10_000_000),
        ("happo-shaped-nodl-nopush-resume15m", 15_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r1", 20_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r2", 25_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r3", 30_000_000),
        ("happo-shaped-nodl-nopush-mixv012-r4", 35_000_000),
    ],
    "v2 cap12": [
        ("happo-shaped-v2", 0),
        ("happo-shaped-v2-cap12-resume10m", 5_000_000),
        ("happo-shaped-v2-cap12-resume15m", 10_000_000),
        ("happo-shaped-v2-cap12-resume20m", 15_000_000),
        ("happo-shaped-v2-cap12-mixv012-r1", 20_000_000),
        ("happo-shaped-v2-cap12-mixv012-r2", 25_000_000),
        ("happo-shaped-v2-cap12-mixv012-r3", 30_000_000),
        ("happo-shaped-v2-cap12-mixv012-r4", 35_000_000),
    ],
    "gru0 no action history": [
        ("happo-rnn-bfs-nodl-nopush-nohist", 0),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-resume20m", 15_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r1", 20_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r2", 25_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r3", 30_000_000),
        ("happo-rnn-bfs-nodl-nopush-nohist-mixv012-r4", 35_000_000),
    ],
    "gru1 one action history": [
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32", 0),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume10m", 5_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume15m", 10_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-resume20m", 15_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r1", 20_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r2", 25_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r3", 30_000_000),
        ("happo-rnn-bfs-nodl-nopush-hist1-thr32-mixv012-r4", 35_000_000),
    ],
}


def max_logged_step(run_dir):
    progress_path = run_dir / "progress.txt"
    max_step = 0
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for pattern in [r"total num timesteps\s+(\d+)/(\d+)", r"^\s*(\d+(?:\.\d+)?)\s*,"]:
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
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_index(output_dir, results_root):
    lines = [
        "# V2 Cap12 Full Comparison",
        "",
        "All curves use stitched 5M segments. The dashed vertical line marks the 20M pretrain/pseudo-mix boundary.",
        "",
        "Metrics:",
    ]
    for key, title in KEY_METRICS:
        lines.append(f"- `{key}`: {title}")
    lines.append("")

    for family_name, segments in FAMILIES.items():
        lines.append(f"## {family_name}")
        for prefix, offset in segments:
            run_dir = latest_run_dir(results_root, prefix)
            if run_dir is None:
                lines.append(f"- {prefix}: missing, offset {offset}")
                continue
            logged_step = max_logged_step(run_dir)
            target = target_steps(run_dir)
            status = "complete" if target and logged_step >= target * 0.95 else "partial"
            lines.append(
                f"- {prefix}: {status}, offset {offset}, local step {logged_step}/{target}, `{run_dir}`"
            )
        lines.append("")
    (output_dir / "run_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/v2-cap12-full-comparison"),
    )
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    series, _ = collect_key_metrics(args.results_root, FAMILIES)
    output_path = args.output_dir / "v2_cap12_vs_baselines_full_key_metrics.png"
    plot_key_metrics(
        series,
        output_path,
        "V2 cap12 vs baseline/v1/GRU: pretrain + pseudo-mix",
        args.smoothing,
    )
    write_index(args.output_dir, args.results_root)
    print(f"Saved plot: {output_path}")
    print(f"Index: {args.output_dir / 'run_index.md'}")


if __name__ == "__main__":
    main()
