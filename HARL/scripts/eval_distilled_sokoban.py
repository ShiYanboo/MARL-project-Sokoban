#!/usr/bin/env python
"""Evaluate distilled Sokoban policies without TensorBoard."""

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HARL_ROOT = REPO_ROOT / "HARL"
if str(HARL_ROOT) not in sys.path:
    sys.path.insert(0, str(HARL_ROOT))

from harl.envs.sokoban.sokoban_env import SokobanEnv  # noqa: E402
from harl.models.policy_models.stochastic_policy import StochasticPolicy  # noqa: E402


METRIC_KEYS = [
    "episode_reward",
    "success",
    "final_boxes_on_target",
    "box_completion_ratio",
    "box_pushes",
    "player_moves",
    "noop_rate",
    "episode_length",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run rollout evaluation for distilled Sokoban model directories."
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        required=True,
        help="Distillation result directories, each containing distill_config.json and models/.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-dir",
        default="plots/distill13-eval",
        help="Directory for eval_metrics.json and plots.",
    )
    return parser.parse_args()


def make_target_env_args(distill_config):
    source_config = distill_config["source_config"]
    env_args = copy.deepcopy(source_config["env_args"])
    distill_args = distill_config["distill_args"]
    target_obs_type = distill_args.get("target_observation_type", "same")
    if target_obs_type == "same":
        target_obs_type = env_args.get("observation_type", "vector")
    env_args["observation_type"] = target_obs_type
    env_args["use_mixed_obs_padding"] = True
    env_args["obs_dim_room"] = int(distill_args.get("target_dim_room", 13))
    env_args["mixed_obs_padding_mode"] = distill_args.get(
        "padding_mode", "random_episode"
    )
    return env_args


def make_model_args(distill_config, target_obs_type):
    source_config = distill_config["source_config"]
    algo_args = source_config["algo_args"]
    model_args = copy.deepcopy(algo_args["model"])
    model_args.update(copy.deepcopy(algo_args["algo"]))
    if target_obs_type == "cnn":
        model_args["cnn_architecture"] = "sokoban"
        model_args["cnn_input_scale"] = 1.0
    return model_args


def load_policies(result_dir, env, model_args, device):
    policies = []
    models_dir = Path(result_dir) / "models"
    for agent_id in range(env.n_agents):
        policy = StochasticPolicy(
            model_args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        actor_path = models_dir / f"actor_agent{agent_id}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        policy.load_state_dict(torch.load(actor_path, map_location=device))
        policy.eval()
        policies.append(policy)
    return policies


def evaluate_one(result_dir, episodes, seed, device, deterministic):
    result_dir = Path(result_dir)
    with (result_dir / "distill_config.json").open("r", encoding="utf-8") as file:
        distill_config = json.load(file)
    env_args = make_target_env_args(distill_config)
    env = SokobanEnv(env_args)
    env.seed(seed)
    target_obs_type = env_args.get("observation_type", "vector")
    model_args = make_model_args(distill_config, target_obs_type)
    policies = load_policies(result_dir, env, model_args, device)

    recurrent_n = int(model_args.get("recurrent_n", 1))
    hidden_size = int(model_args["hidden_sizes"][-1])

    episode_rows = []
    for episode_index in range(episodes):
        obs, _, avail_actions = env.reset()
        rnn_states = [
            np.zeros((1, recurrent_n, hidden_size), dtype=np.float32)
            for _ in range(env.n_agents)
        ]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)]
        done = False
        ep_reward = 0.0
        ep_len = 0
        box_pushes = 0
        player_moves = 0
        noops = 0
        final_info = {}
        while not done:
            actions = []
            for agent_id, policy in enumerate(policies):
                with torch.no_grad():
                    action, next_rnn = policy(
                        np.expand_dims(obs[agent_id], axis=0),
                        rnn_states[agent_id],
                        masks[agent_id],
                        np.expand_dims(np.asarray(avail_actions[agent_id]), axis=0),
                        deterministic=deterministic,
                    )[0::2]
                actions.append(int(action.cpu().numpy().reshape(-1)[0]))
                rnn_states[agent_id] = next_rnn.cpu().numpy()
            obs, _, rewards, dones, infos, avail_actions = env.step(actions)
            info = infos[0]
            reward = float(np.mean(rewards))
            done = bool(np.all(dones))
            ep_reward += reward
            ep_len += 1
            box_pushes += int(info.get("action_moved_box", False))
            player_moves += int(info.get("action_moved_player", False))
            noops += int(info.get("noop_executed", False))
            final_info = info
            if done:
                masks = [
                    np.zeros((1, 1), dtype=np.float32) for _ in range(env.n_agents)
                ]
            else:
                masks = [
                    np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)
                ]

        episode_rows.append(
            {
                "episode": episode_index + 1,
                "episode_reward": ep_reward,
                "success": float(final_info.get("success", False)),
                "final_boxes_on_target": float(final_info.get("boxes_on_target", 0)),
                "box_completion_ratio": float(
                    final_info.get("box_completion_ratio", 0.0)
                ),
                "box_pushes": float(box_pushes),
                "player_moves": float(player_moves),
                "noop_rate": float(noops) / max(ep_len, 1),
                "episode_length": float(ep_len),
            }
        )
    env.close()

    aggregate = {}
    for key in METRIC_KEYS:
        values = np.array([row[key] for row in episode_rows], dtype=np.float64)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "result_dir": str(result_dir),
        "name": result_dir.name,
        "episodes": episode_rows,
        "aggregate": aggregate,
    }


def plot_metrics(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [result["name"] for result in results]
    short_names = [name.replace("distill13-", "") for name in names]
    plot_specs = [
        ("success", "success rate"),
        ("box_completion_ratio", "box completion ratio"),
        ("final_boxes_on_target", "final boxes on target"),
        ("episode_reward", "episode reward"),
        ("box_pushes", "box pushes"),
        ("noop_rate", "noop rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for axis, (key, title) in zip(axes, plot_specs):
        means = [result["aggregate"][key]["mean"] for result in results]
        stds = [result["aggregate"][key]["std"] for result in results]
        axis.bar(range(len(results)), means, yerr=stds, capsize=3)
        axis.set_title(title)
        axis.set_xticks(range(len(results)))
        axis.set_xticklabels(short_names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "distill_eval_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_one(path, args.episodes, args.seed + index * 1000, device, args.deterministic)
        for index, path in enumerate(args.result_dirs)
    ]
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    plot_metrics(results, output_dir)
    for result in results:
        aggregate = result["aggregate"]
        print(
            f"{result['name']}: "
            f"success={aggregate['success']['mean']:.3f}, "
            f"boxes={aggregate['final_boxes_on_target']['mean']:.3f}, "
            f"completion={aggregate['box_completion_ratio']['mean']:.3f}, "
            f"reward={aggregate['episode_reward']['mean']:.3f}"
        )
    print(f"Wrote {output_dir / 'eval_metrics.json'}")
    print(f"Wrote {output_dir / 'distill_eval_summary.png'}")


if __name__ == "__main__":
    main()
