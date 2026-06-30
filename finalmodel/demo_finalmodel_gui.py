#!/usr/bin/env python
"""Run a 3x3 graphical demo for the archived final Sokoban model."""

import argparse
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: torch. Run this demo through ./demo_finalmodel_gui.sh "
        "or activate the harl-sokoban conda environment first."
    ) from error

try:
    import gym_notices.notices as gym_notices

    gym_notices.notices.clear()
except Exception:
    pass

from eval_finalmodel import (  # noqa: E402
    load_config,
    load_policies,
    make_env_args,
    make_model_args,
    resolve_device,
)
from harl.envs.sokoban.sokoban_env import SokobanEnv  # noqa: E402


LOCAL_ACTIONS = {
    0: "noop",
    1: "push up",
    2: "push down",
    3: "push left",
    4: "push right",
    5: "move up",
    6: "move down",
    7: "move left",
    8: "move right",
}


@dataclass
class SceneState:
    index: int
    scenario: str
    seed: int
    env: SokobanEnv
    recurrent_n: int
    hidden_size: int
    obs: list = field(default_factory=list)
    avail_actions: list = field(default_factory=list)
    rnn_states: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    episode: int = 0
    step: int = 0
    reward: float = 0.0
    done: bool = False
    last_info: dict = field(default_factory=dict)
    last_actions: list = field(default_factory=list)

    def reset(self):
        self.obs, _, self.avail_actions = self.env.reset()
        self.rnn_states = [
            np.zeros((1, self.recurrent_n, self.hidden_size), dtype=np.float32)
            for _ in range(self.env.n_agents)
        ]
        self.masks = [
            np.ones((1, 1), dtype=np.float32) for _ in range(self.env.n_agents)
        ]
        self.episode += 1
        self.step = 0
        self.reward = 0.0
        self.done = False
        self.last_info = {}
        self.last_actions = []

    def render(self):
        return self.env.render(mode="rgb_array")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load finalmodel/models/actor_agent*.pt and show nine synchronized "
            "TwoPlayer-Sokoban inference scenes in a 3x3 grid."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=THIS_DIR,
        help="Directory containing config.json and models/.",
    )
    parser.add_argument(
        "--scenarios",
        default="config",
        help=(
            "Comma-separated scenarios, or 'config' to cycle the scenario_pool "
            "stored in finalmodel/config.json."
        ),
    )
    parser.add_argument("--num-scenes", type=int, default=9)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of environment steps to run. 0 means run until the window closes.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a GUI window. Useful with --save-gif for verification.",
    )
    parser.add_argument(
        "--save-gif",
        type=Path,
        default=None,
        help="Optional path for saving the rendered demo as a GIF.",
    )
    return parser.parse_args()


def import_matplotlib(headless):
    warnings.filterwarnings("ignore", message=".*parseString.*")
    warnings.filterwarnings("ignore", message=".*resetCache.*")
    warnings.filterwarnings("ignore", message=".*enablePackrat.*")

    import matplotlib

    if headless:
        matplotlib.use("Agg")
    elif sys.platform.startswith("linux") and not _has_display():
        raise SystemExit(
            "No graphical display was detected. Run from a desktop terminal, "
            "or use --headless --save-gif demo.gif."
        )

    import matplotlib.pyplot as plt

    return plt


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def parse_scenarios(value, config, num_scenes):
    if value == "config":
        pool = config.get("env_args", {}).get("scenario_pool", "")
        scenarios = [item.strip() for item in pool.split(",") if item.strip()]
        if not scenarios:
            scenarios = [
                config.get("env_args", {}).get("scenario", "TwoPlayer-Sokoban-v0")
            ]
    else:
        scenarios = [item.strip() for item in value.split(",") if item.strip()]

    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if num_scenes <= 0:
        raise ValueError("--num-scenes must be positive.")

    return [scenarios[i % len(scenarios)] for i in range(num_scenes)]


def resolve_demo_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return resolve_device(device_arg)


def build_scenes(config, scenarios, model_args):
    recurrent_n = int(model_args.get("recurrent_n", 1))
    hidden_size = int(model_args["hidden_sizes"][-1])
    scenes = []

    for index, scenario in enumerate(scenarios):
        env_args = make_env_args(config, scenario, "")
        env = SokobanEnv(env_args)
        scenes.append(
            SceneState(
                index=index,
                scenario=scenario,
                seed=0,
                env=env,
                recurrent_n=recurrent_n,
                hidden_size=hidden_size,
            )
        )

    return scenes


def seed_and_reset_scenes(scenes, base_seed):
    for scene in scenes:
        scene.seed = base_seed + scene.index
        scene.env.seed(scene.seed)
        scene.reset()


@torch.no_grad()
def advance_scene(scene, policies, deterministic):
    if scene.done:
        scene.reset()

    actions = []
    for agent_id, policy in enumerate(policies):
        action, _, next_rnn = policy(
            np.expand_dims(scene.obs[agent_id], axis=0),
            scene.rnn_states[agent_id],
            scene.masks[agent_id],
            np.expand_dims(np.asarray(scene.avail_actions[agent_id]), axis=0),
            deterministic=deterministic,
        )
        actions.append(int(action.cpu().numpy().reshape(-1)[0]))
        scene.rnn_states[agent_id] = next_rnn.cpu().numpy()

    obs, _, rewards, dones, infos, avail_actions = scene.env.step(actions)
    done = bool(np.all(dones))
    info = infos[0]
    scene.obs = obs
    scene.avail_actions = avail_actions
    scene.done = done
    scene.last_info = info
    scene.last_actions = actions
    scene.step += 1
    scene.reward += float(np.mean(rewards))

    mask_value = 0.0 if done else 1.0
    scene.masks = [
        np.full((1, 1), mask_value, dtype=np.float32)
        for _ in range(scene.env.n_agents)
    ]


def create_figure(plt, scenes):
    cols = int(math.ceil(math.sqrt(len(scenes))))
    rows = int(math.ceil(len(scenes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.asarray(axes).reshape(-1)
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.92,
        wspace=0.08,
        hspace=0.35,
    )
    try:
        fig.canvas.manager.set_window_title("Final Model Sokoban Demo")
    except Exception:
        pass
    fig.suptitle("Final GRU0 model inference demo", fontsize=13)

    artists = []
    for ax, scene in zip(axes, scenes):
        image = ax.imshow(scene.render(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(scene_title(scene), fontsize=8)
        artists.append(image)

    for ax in axes[len(scenes):]:
        ax.axis("off")

    return fig, axes, artists


def update_figure(fig, axes, artists, scenes):
    for ax, artist, scene in zip(axes, artists, scenes):
        artist.set_data(scene.render())
        ax.set_title(scene_title(scene), fontsize=8)
    fig.canvas.draw_idle()


def scene_title(scene):
    info = scene.last_info
    boxes = info.get("boxes_on_target", getattr(scene.env.env, "boxes_on_target", 0))
    total_boxes = info.get("num_boxes", getattr(scene.env.env, "num_boxes", 0))
    status = (
        "success" if info.get("success", False) else ("done" if scene.done else "running")
    )
    chosen = info.get("chosen_agent", "-")
    if scene.last_actions:
        local_action = scene.last_actions[int(chosen)] if chosen in {0, 1} else 0
        action_text = f"A{chosen} {LOCAL_ACTIONS.get(local_action, local_action)}"
    else:
        action_text = "initial"
    return (
        f"#{scene.index + 1} {scene.scenario}\n"
        f"ep {scene.episode} step {scene.step} boxes {boxes}/{total_boxes} {status}\n"
        f"{action_text} reward {scene.reward:.2f}"
    )


def capture_figure(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = rgba.reshape((height, width, 4))
    return rgba[:, :, :3].copy()


def pause_or_wait(plt, fig, seconds, headless):
    if headless:
        return True

    deadline = time.time() + max(seconds, 0.0)
    while time.time() < deadline:
        if not plt.fignum_exists(fig.number):
            return False
        plt.pause(min(0.05, max(deadline - time.time(), 0.001)))
    return plt.fignum_exists(fig.number)


def save_gif(path, frames, interval):
    import imageio.v2 as imageio

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=max(interval, 0.01))


def main():
    args = parse_args()
    if args.headless and args.steps <= 0:
        raise SystemExit("--headless requires --steps so the demo can terminate.")

    try:
        import gym

        gym.logger.set_level(40)
    except Exception:
        pass

    plt = import_matplotlib(args.headless)
    model_dir = args.model_dir.resolve()
    config = load_config(model_dir)
    model_args = make_model_args(config)
    scenarios = parse_scenarios(args.scenarios, config, args.num_scenes)
    device = resolve_demo_device(args.device)

    scenes = build_scenes(config, scenarios, model_args)
    try:
        policies = load_policies(model_dir, scenes[0].env, model_args, device)
        seed_and_reset_scenes(scenes, args.seed)
        fig, axes, artists = create_figure(plt, scenes)
        update_figure(fig, axes, artists, scenes)
        frames = []
        if args.save_gif is not None:
            frames.append(capture_figure(fig))

        if not args.headless:
            plt.show(block=False)

        steps_run = 0
        while args.steps <= 0 or steps_run < args.steps:
            if not pause_or_wait(plt, fig, args.interval, args.headless):
                break
            for scene in scenes:
                advance_scene(scene, policies, args.deterministic)
            steps_run += 1
            update_figure(fig, axes, artists, scenes)
            if args.save_gif is not None:
                frames.append(capture_figure(fig))

        if args.save_gif is not None:
            save_gif(args.save_gif, frames, args.interval)
            print(f"wrote GIF: {args.save_gif}")
    finally:
        for scene in locals().get("scenes", []):
            scene.env.close()


if __name__ == "__main__":
    main()
