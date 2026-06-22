import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_check(name, extra_args):
    cmd = [
        sys.executable,
        "examples/train.py",
        *extra_args,
    ]
    print(f"[verify] running {name}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    run_check(
        "happo",
        [
            "--algo",
            "happo",
            "--env",
            "sokoban",
            "--exp_name",
            "verify_happo",
            "--scenario",
            "TwoPlayer-Sokoban-v0",
            "--n_rollout_threads",
            "1",
            "--n_eval_rollout_threads",
            "1",
            "--episode_length",
            "20",
            "--num_env_steps",
            "40",
            "--control_mode",
            "turn_based",
            "--max_steps",
            "40",
            "--dim_room",
            "7",
            "--num_boxes",
            "1",
            "--eval_interval",
            "1",
            "--eval_episodes",
            "1",
            "--log_interval",
            "1",
            "--cuda",
            "False",
            "--use_eval",
            "True",
        ],
    )
    run_check(
        "hasac",
        [
            "--algo",
            "hasac",
            "--env",
            "sokoban",
            "--exp_name",
            "verify_hasac",
            "--scenario",
            "TwoPlayer-Sokoban-v0",
            "--n_rollout_threads",
            "1",
            "--n_eval_rollout_threads",
            "1",
            "--num_env_steps",
            "80",
            "--max_steps",
            "40",
            "--dim_room",
            "7",
            "--num_boxes",
            "1",
            "--warmup_steps",
            "20",
            "--train_interval",
            "10",
            "--update_per_train",
            "1",
            "--eval_interval",
            "20",
            "--eval_episodes",
            "1",
            "--batch_size",
            "8",
            "--buffer_size",
            "200",
            "--n_step",
            "2",
            "--cuda",
            "False",
            "--use_eval",
            "True",
        ],
    )


if __name__ == "__main__":
    main()
