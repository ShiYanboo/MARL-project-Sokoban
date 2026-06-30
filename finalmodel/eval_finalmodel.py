#!/usr/bin/env python
"""Evaluate the archived final GRU0 model on TwoPlayer-Sokoban-v0."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HARL_ROOT = REPO_ROOT / "HARL"
if str(HARL_ROOT) not in sys.path:
    sys.path.insert(0, str(HARL_ROOT))

from harl.envs.sokoban.sokoban_env import SokobanEnv  # noqa: E402
from harl.models.policy_models.stochastic_policy import StochasticPolicy  # noqa: E402


METRIC_KEYS = [
    "episode_reward",
    "episode_base_reward",
    "episode_shaping_reward",
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
        description="Evaluate finalmodel/ GRU0 checkpoint on v0 or a scenario pool."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing config.json and models/.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=50000,
        help=(
            "Sokoban seed for the first episode. The default matches HARL's "
            "eval seed rule for training seed=1 and rank=0: seed * 50000."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenario", default="TwoPlayer-Sokoban-v0")
    parser.add_argument(
        "--scenario_pool",
        default="",
        help=(
            "Optional comma-separated scenario pool. Use 'config' to reuse the "
            "scenario_pool stored in finalmodel/config.json. Empty means v0-only."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <model-dir>/eval_v0_metrics.json.",
    )
    return parser.parse_args()


def load_config(model_dir):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_env_args(config, scenario, scenario_pool):
    env_args = copy.deepcopy(config["env_args"])
    if scenario_pool == "config":
        scenario_pool = env_args.get("scenario_pool", "")
    if scenario_pool:
        pool = [item.strip() for item in scenario_pool.split(",") if item.strip()]
        if not pool:
            raise ValueError("--scenario_pool did not contain any valid scenarios.")
        env_args["scenario"] = pool[0]
        env_args["scenario_pool"] = ",".join(pool)
    else:
        env_args["scenario"] = scenario
        env_args["scenario_pool"] = ""
    env_args["dim_room"] = 7
    env_args["num_boxes"] = 2
    env_args["use_mixed_obs_padding"] = False
    env_args["obs_dim_room"] = None
    env_args["mixed_obs_padding_mode"] = "top_left"
    return env_args


def make_model_args(config):
    algo_args = config["algo_args"]
    model_args = copy.deepcopy(algo_args["model"])
    model_args.update(copy.deepcopy(algo_args["algo"]))
    return model_args


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_policies(model_dir, env, model_args, device):
    policies = []
    models_dir = model_dir / "models"
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


def aggregate(rows):
    summary = {}
    for key in METRIC_KEYS:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def aggregate_by_scenario(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    return {
        scenario_name: aggregate(scenario_rows)
        for scenario_name, scenario_rows in grouped.items()
    }


@torch.no_grad()
def evaluate(model_dir, episodes, seed, device, scenario, scenario_pool, deterministic):
    config = load_config(model_dir)
    env_args = make_env_args(config, scenario, scenario_pool)
    model_args = make_model_args(config)

    env = SokobanEnv(env_args)
    env.seed(seed)
    policies = load_policies(model_dir, env, model_args, device)

    recurrent_n = int(model_args.get("recurrent_n", 1))
    hidden_size = int(model_args["hidden_sizes"][-1])

    rows = []
    for episode_index in range(episodes):
        obs, _, avail_actions = env.reset()
        rnn_states = [
            np.zeros((1, recurrent_n, hidden_size), dtype=np.float32)
            for _ in range(env.n_agents)
        ]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in range(env.n_agents)]

        done = False
        ep_reward = 0.0
        ep_base_reward = 0.0
        ep_shaping_reward = 0.0
        ep_len = 0
        box_pushes = 0
        player_moves = 0
        noops = 0
        final_info = {}

        while not done:
            actions = []
            for agent_id, policy in enumerate(policies):
                action, _, next_rnn = policy(
                    np.expand_dims(obs[agent_id], axis=0),
                    rnn_states[agent_id],
                    masks[agent_id],
                    np.expand_dims(np.asarray(avail_actions[agent_id]), axis=0),
                    deterministic=deterministic,
                )
                actions.append(int(action.cpu().numpy().reshape(-1)[0]))
                rnn_states[agent_id] = next_rnn.cpu().numpy()

            obs, _, rewards, dones, infos, avail_actions = env.step(actions)
            info = infos[0]
            done = bool(np.all(dones))
            step_reward = float(np.mean(rewards))

            ep_reward += step_reward
            ep_base_reward += float(info.get("base_reward", 0.0))
            ep_shaping_reward += float(info.get("shaping_reward", 0.0))
            ep_len += 1
            box_pushes += int(info.get("action_moved_box", False))
            player_moves += int(info.get("action_moved_player", False))
            noops += int(info.get("noop_executed", False))
            final_info = info

            mask_value = 0.0 if done else 1.0
            masks = [
                np.full((1, 1), mask_value, dtype=np.float32)
                for _ in range(env.n_agents)
            ]

        rows.append(
            {
                "episode": episode_index + 1,
                "scenario": str(final_info.get("scenario", env.scenario)),
                "episode_reward": ep_reward,
                "episode_base_reward": ep_base_reward,
                "episode_shaping_reward": ep_shaping_reward,
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
    return {
        "model_dir": str(model_dir),
        "scenario": scenario,
        "scenario_pool": env_args.get("scenario_pool", ""),
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "aggregate": aggregate(rows),
        "aggregate_by_scenario": aggregate_by_scenario(rows),
        "episode_rows": rows,
    }


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_json = args.output_json or (model_dir / "eval_v0_metrics.json")
    device = resolve_device(args.device)

    result = evaluate(
        model_dir=model_dir,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
        scenario=args.scenario,
        scenario_pool=args.scenario_pool,
        deterministic=args.deterministic,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    agg = result["aggregate"]
    print(f"model_dir: {model_dir}")
    print(f"scenario: {args.scenario}")
    print(f"scenario_pool: {result['scenario_pool'] or '(none)'}")
    print(f"episodes: {args.episodes}")
    print(f"device: {device}")
    print(f"success_rate: {agg['success']['mean']:.4f}")
    print(f"box_completion_ratio: {agg['box_completion_ratio']['mean']:.4f}")
    print(f"final_boxes_on_target: {agg['final_boxes_on_target']['mean']:.4f}")
    print(f"episode_reward: {agg['episode_reward']['mean']:.4f}")
    print(f"episode_base_reward: {agg['episode_base_reward']['mean']:.4f}")
    print(f"episode_shaping_reward: {agg['episode_shaping_reward']['mean']:.4f}")
    print(f"episode_length: {agg['episode_length']['mean']:.4f}")
    if result["aggregate_by_scenario"]:
        print("per_scenario:")
        for scenario_name, scenario_agg in sorted(result["aggregate_by_scenario"].items()):
            count = sum(
                1
                for row in result["episode_rows"]
                if row["scenario"] == scenario_name
            )
            print(
                f"  {scenario_name}: "
                f"n={count}, "
                f"success={scenario_agg['success']['mean']:.4f}, "
                f"box_completion={scenario_agg['box_completion_ratio']['mean']:.4f}"
            )
    print(f"wrote: {output_json}")


if __name__ == "__main__":
    main()
