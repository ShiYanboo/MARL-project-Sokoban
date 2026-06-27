#!/usr/bin/env python
"""Distill a trained Sokoban policy into a larger padded observation canvas."""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete

REPO_ROOT = Path(__file__).resolve().parents[2]
HARL_ROOT = REPO_ROOT / "HARL"
if str(HARL_ROOT) not in sys.path:
    sys.path.insert(0, str(HARL_ROOT))

from harl.envs.sokoban.sokoban_env import SokobanEnv  # noqa: E402
from harl.models.policy_models.stochastic_policy import StochasticPolicy  # noqa: E402
from harl.models.value_function_models.v_net import VNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Imitate a trained 7x7/10x10 Sokoban policy on a larger padded canvas. "
            "The saved output is a HARL-compatible models/ directory."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Source HARL models directory.")
    parser.add_argument(
        "--student-model-dir",
        default=None,
        help=(
            "Optional existing distilled student models directory to continue from. "
            "If omitted, initialize the student from the teacher where shapes permit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. A models/ subdirectory is created inside it.",
    )
    parser.add_argument("--target-dim-room", type=int, default=13)
    parser.add_argument(
        "--target-observation-type",
        choices=["same", "vector", "cnn"],
        default="same",
        help="Student observation type. 'same' preserves the teacher type.",
    )
    parser.add_argument(
        "--padding-mode",
        choices=["top_left", "center", "random_episode"],
        default="random_episode",
        help="How to place smaller source boards inside the larger student canvas.",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--batch-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rollout-action-mode",
        choices=["deterministic", "sample"],
        default="deterministic",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Uses CPU automatically if CUDA is unavailable.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional torch.set_num_threads value for CPU-side work.",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--copy-value-normalizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_run_config(model_dir):
    model_dir = Path(model_dir).resolve()
    for parent in [model_dir, *model_dir.parents]:
        config_path = parent / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as file:
                return json.load(file), config_path
    raise FileNotFoundError(f"Could not find config.json above {model_dir}")


def to_room_shape(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if value is None:
        raise ValueError("Source env config has no dim_room; pass a model with saved config.")
    raise ValueError(f"Unsupported dim_room value: {value!r}")


def make_source_env_args(env_args, seed):
    args = copy.deepcopy(env_args)
    args["use_mixed_obs_padding"] = False
    args.pop("obs_dim_room", None)
    args.pop("mixed_obs_padding_mode", None)
    return args


def make_target_env_args(env_args, target_dim, target_observation_type, padding_mode):
    args = copy.deepcopy(env_args)
    if target_observation_type != "same":
        args["observation_type"] = target_observation_type
    args["use_mixed_obs_padding"] = True
    args["obs_dim_room"] = target_dim
    args["mixed_obs_padding_mode"] = padding_mode
    return args


def make_model_args(algo_args, target_observation_type):
    model_args = copy.deepcopy(algo_args["model"])
    model_args.update(copy.deepcopy(algo_args["algo"]))
    if target_observation_type == "cnn":
        model_args["cnn_architecture"] = "sokoban"
        model_args["cnn_input_scale"] = 1.0
    return model_args


def load_policy(path, model_args, obs_space, action_space, device):
    policy = StochasticPolicy(model_args, obs_space, action_space, device=device)
    state = torch.load(path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def load_value(path, model_args, share_obs_space, device):
    value = VNet(model_args, share_obs_space, device=device)
    state = torch.load(path, map_location=device)
    value.load_state_dict(state)
    value.eval()
    return value


def load_student_models(student_model_dir, target_actors, target_critic, device):
    student_model_dir = Path(student_model_dir)
    for agent_id, actor in enumerate(target_actors):
        actor_path = student_model_dir / f"actor_agent{agent_id}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing student actor checkpoint: {actor_path}")
        actor.load_state_dict(torch.load(actor_path, map_location=device))
    critic_path = student_model_dir / "critic_agent.pt"
    if not critic_path.exists():
        raise FileNotFoundError(f"Missing student critic checkpoint: {critic_path}")
    target_critic.load_state_dict(torch.load(critic_path, map_location=device))


def create_policy(model_args, obs_space, action_space, device):
    policy = StochasticPolicy(model_args, obs_space, action_space, device=device)
    policy.train()
    return policy


def create_value(model_args, share_obs_space, device):
    value = VNet(model_args, share_obs_space, device=device)
    value.train()
    return value


def fixed_offset(source_shape, target_shape, mode, rng):
    max_row = target_shape[0] - source_shape[0]
    max_col = target_shape[1] - source_shape[1]
    if max_row < 0 or max_col < 0:
        raise ValueError(f"Cannot fit source board {source_shape} into {target_shape}.")
    if mode == "top_left":
        return (0, 0)
    if mode == "center":
        return (max_row // 2, max_col // 2)
    return (
        int(rng.integers(0, max_row + 1)) if max_row > 0 else 0,
        int(rng.integers(0, max_col + 1)) if max_col > 0 else 0,
    )


def split_vector_obs(obs, room_shape, action_history_len, n_agents, action_dim):
    obs = np.asarray(obs, dtype=np.float32)
    board_size = int(room_shape[0] * room_shape[1])
    board_end = 6 * board_size
    history_len = int(action_history_len * n_agents * action_dim)
    channels = obs[:board_end].reshape(6, room_shape[0], room_shape[1])
    extras = obs[board_end : board_end + 3]
    history = obs[board_end + 3 : board_end + 3 + history_len]
    return channels, extras, history


def split_cnn_obs(obs, action_history_len, n_agents, action_dim):
    obs = np.asarray(obs, dtype=np.float32)
    history_len = int(action_history_len * n_agents * action_dim)
    channels = obs[:6]
    extras = np.array([obs[6].mean(), obs[7].mean(), obs[8].mean()], dtype=np.float32)
    history = np.array(
        [obs[9 + index].mean() for index in range(history_len)], dtype=np.float32
    )
    return channels, extras, history


def pad_channels(channels, target_shape, offset):
    padded = np.zeros((channels.shape[0], target_shape[0], target_shape[1]), dtype=np.float32)
    padded[0].fill(1.0)
    rows, cols = channels.shape[1:]
    row_offset, col_offset = offset
    padded[
        :, row_offset : row_offset + rows, col_offset : col_offset + cols
    ] = channels.astype(np.float32)
    return padded


def build_target_obs(
    source_obs,
    source_observation_type,
    target_observation_type,
    source_shape,
    target_shape,
    offset,
    action_history_len,
    n_agents,
    action_dim,
):
    if source_observation_type == "cnn":
        channels, extras, history = split_cnn_obs(
            source_obs, action_history_len, n_agents, action_dim
        )
    else:
        channels, extras, history = split_vector_obs(
            source_obs, source_shape, action_history_len, n_agents, action_dim
        )
    padded = pad_channels(channels, target_shape, offset)
    if target_observation_type == "cnn":
        constant_maps = [
            np.full(target_shape, value, dtype=np.float32)
            for value in np.concatenate([extras, history], axis=0)
        ]
        return np.concatenate([padded, np.stack(constant_maps, axis=0)], axis=0)
    return np.concatenate(
        [padded.reshape(-1), extras.astype(np.float32), history.astype(np.float32)],
        axis=0,
    ).astype(np.float32)


def mapped_vector_indices(source_shape, target_shape, offset, history_len):
    src_indices = []
    tgt_indices = []
    source_board = source_shape[0] * source_shape[1]
    target_board = target_shape[0] * target_shape[1]
    row_offset, col_offset = offset
    for channel in range(6):
        for row in range(source_shape[0]):
            for col in range(source_shape[1]):
                src_indices.append(channel * source_board + row * source_shape[1] + col)
                tgt_indices.append(
                    channel * target_board
                    + (row + row_offset) * target_shape[1]
                    + col
                    + col_offset
                )
    src_extra = 6 * source_board
    tgt_extra = 6 * target_board
    for index in range(3 + history_len):
        src_indices.append(src_extra + index)
        tgt_indices.append(tgt_extra + index)
    return torch.as_tensor(src_indices, dtype=torch.long), torch.as_tensor(
        tgt_indices, dtype=torch.long
    )


def inflate_vector_state_dict(target_module, source_state, source_shape, target_shape, history_len):
    target_state = target_module.state_dict()
    center_offset = (
        (target_shape[0] - source_shape[0]) // 2,
        (target_shape[1] - source_shape[1]) // 2,
    )
    src_idx, tgt_idx = mapped_vector_indices(
        source_shape, target_shape, center_offset, history_len
    )
    for key, value in source_state.items():
        if key not in target_state:
            continue
        if target_state[key].shape == value.shape:
            target_state[key] = value.clone()
            continue
        if key == "base.mlp.fc.0.weight" and target_state[key].ndim == 2:
            src_idx_device = src_idx.to(value.device)
            tgt_idx_device = tgt_idx.to(target_state[key].device)
            target_state[key][:, tgt_idx_device] = value[:, src_idx_device]
        elif key in {"base.feature_norm.weight", "base.feature_norm.bias"}:
            src_idx_device = src_idx.to(value.device)
            tgt_idx_device = tgt_idx.to(target_state[key].device)
            target_state[key][tgt_idx_device] = value[src_idx_device]
    target_module.load_state_dict(target_state)


def copy_compatible_state_dict(target_module, source_state):
    target_state = target_module.state_dict()
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.clone()
    target_module.load_state_dict(target_state)


def policy_distribution(policy, obs, rnn_states, masks, available_actions):
    features = policy.base(obs)
    if policy.use_naive_recurrent_policy or policy.use_recurrent_policy:
        features, rnn_states = policy.rnn(features, rnn_states, masks)
    distribution = policy.act.action_out(features, available_actions)
    return distribution, rnn_states


@torch.no_grad()
def teacher_step(
    source_actors,
    source_critic,
    obs,
    share_obs,
    available_actions,
    actor_rnn_states,
    critic_rnn_state,
    masks,
    critic_mask,
    action_mode,
    device,
):
    actions = []
    probs = []
    for agent_id, actor in enumerate(source_actors):
        obs_tensor = torch.as_tensor(obs[agent_id][None], dtype=torch.float32, device=device)
        rnn_tensor = torch.as_tensor(
            actor_rnn_states[agent_id : agent_id + 1], dtype=torch.float32, device=device
        )
        mask_tensor = torch.as_tensor(masks[agent_id : agent_id + 1], dtype=torch.float32, device=device)
        avail_tensor = torch.as_tensor(
            np.asarray(available_actions[agent_id], dtype=np.float32)[None],
            dtype=torch.float32,
            device=device,
        )
        dist, next_rnn = policy_distribution(
            actor, obs_tensor, rnn_tensor, mask_tensor, avail_tensor
        )
        action = dist.mode() if action_mode == "deterministic" else dist.sample()
        actions.append([int(action.item())])
        probs.append(dist.probs.squeeze(0).detach().cpu().numpy().astype(np.float32))
        actor_rnn_states[agent_id] = next_rnn.squeeze(0).detach().cpu().numpy()

    share_tensor = torch.as_tensor(share_obs[0][None], dtype=torch.float32, device=device)
    critic_rnn_tensor = torch.as_tensor(critic_rnn_state, dtype=torch.float32, device=device)
    critic_mask_tensor = torch.as_tensor(critic_mask, dtype=torch.float32, device=device)
    value, next_critic_rnn = source_critic(
        share_tensor, critic_rnn_tensor, critic_mask_tensor
    )
    return (
        np.asarray(actions, dtype=np.int64),
        np.stack(probs, axis=0),
        value.squeeze(0).detach().cpu().numpy().astype(np.float32),
        actor_rnn_states,
        next_critic_rnn.detach().cpu().numpy(),
    )


def train_on_episode(
    target_actors,
    target_critic,
    actor_optimizers,
    critic_optimizer,
    episode,
    model_args,
    temperature,
    value_loss_coef,
    device,
):
    actor_losses = []
    for agent_id, actor in enumerate(target_actors):
        obs = torch.as_tensor(
            np.stack(episode["target_obs"][agent_id], axis=0),
            dtype=torch.float32,
            device=device,
        )
        available_actions = torch.as_tensor(
            np.stack(episode["available_actions"][agent_id], axis=0),
            dtype=torch.float32,
            device=device,
        )
        teacher_probs = torch.as_tensor(
            np.stack(episode["teacher_probs"][agent_id], axis=0),
            dtype=torch.float32,
            device=device,
        )
        masks = torch.ones((obs.shape[0], 1), dtype=torch.float32, device=device)
        rnn_states = torch.zeros(
            (1, model_args["recurrent_n"], model_args["hidden_sizes"][-1]),
            dtype=torch.float32,
            device=device,
        )
        dist, _ = policy_distribution(actor, obs, rnn_states, masks, available_actions)
        student_log_probs = F.log_softmax(dist.logits / temperature, dim=-1)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        actor_optimizers[agent_id].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizers[agent_id].step()
        actor_losses.append(float(loss.item()))

    share_obs = torch.as_tensor(
        np.stack(episode["target_share_obs"], axis=0), dtype=torch.float32, device=device
    )
    teacher_values = torch.as_tensor(
        np.stack(episode["teacher_values"], axis=0), dtype=torch.float32, device=device
    )
    critic_masks = torch.ones((share_obs.shape[0], 1), dtype=torch.float32, device=device)
    critic_rnn = torch.zeros(
        (1, model_args["recurrent_n"], model_args["hidden_sizes"][-1]),
        dtype=torch.float32,
        device=device,
    )
    values, _ = target_critic(share_obs, critic_rnn, critic_masks)
    value_loss = F.mse_loss(values, teacher_values)
    critic_optimizer.zero_grad()
    (value_loss_coef * value_loss).backward()
    torch.nn.utils.clip_grad_norm_(target_critic.parameters(), 10.0)
    critic_optimizer.step()

    return float(np.mean(actor_losses)), float(value_loss.item())


def save_models(
    output_dir,
    target_actors,
    target_critic,
    model_dir,
    run_config,
    distill_args,
    copy_value_normalizer,
):
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, actor in enumerate(target_actors):
        torch.save(actor.state_dict(), models_dir / f"actor_agent{agent_id}.pt")
    torch.save(target_critic.state_dict(), models_dir / "critic_agent.pt")
    source_normalizer = Path(model_dir) / "value_normalizer.pt"
    if copy_value_normalizer and source_normalizer.exists():
        shutil.copy2(source_normalizer, models_dir / "value_normalizer.pt")
    with (output_dir / "distill_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_model_dir": str(model_dir),
                "source_config": run_config,
                "distill_args": vars(distill_args),
                "models_dir": str(models_dir),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return models_dir


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    model_dir = Path(args.model_dir).resolve()
    run_config, config_path = load_run_config(model_dir)
    algo_args = run_config["algo_args"]
    source_env_args = make_source_env_args(run_config["env_args"], args.seed)
    source_observation_type = source_env_args.get("observation_type", "vector")
    target_observation_type = (
        source_observation_type
        if args.target_observation_type == "same"
        else args.target_observation_type
    )
    target_env_args = make_target_env_args(
        run_config["env_args"],
        args.target_dim_room,
        target_observation_type,
        "top_left",
    )

    source_env = SokobanEnv(source_env_args)
    source_env.seed(args.seed)
    target_env = SokobanEnv(target_env_args)
    target_env.seed(args.seed)
    n_agents = source_env.n_agents
    action_dim = source_env.action_space[0].n
    action_history_len = int(source_env_args.get("action_history_len", 0) or 0)
    source_shape = tuple(source_env.env.room_state.shape)
    target_shape = to_room_shape(args.target_dim_room)

    source_model_args = make_model_args(algo_args, source_observation_type)
    target_model_args = make_model_args(algo_args, target_observation_type)
    source_obs_space = source_env.observation_space[0]
    source_share_space = source_env.share_observation_space[0]
    target_obs_space = target_env.observation_space[0]
    target_share_space = target_env.share_observation_space[0]
    action_space = Discrete(action_dim)

    source_actors = [
        load_policy(
            model_dir / f"actor_agent{agent_id}.pt",
            source_model_args,
            source_obs_space,
            action_space,
            device,
        )
        for agent_id in range(n_agents)
    ]
    source_critic = load_value(
        model_dir / "critic_agent.pt", source_model_args, source_share_space, device
    )
    target_actors = [
        create_policy(target_model_args, target_obs_space, action_space, device)
        for _ in range(n_agents)
    ]
    target_critic = create_value(target_model_args, target_share_space, device)

    history_len = action_history_len * n_agents * action_dim
    if args.student_model_dir:
        load_student_models(args.student_model_dir, target_actors, target_critic, device)
    else:
        if source_observation_type == target_observation_type == "vector":
            for agent_id, actor in enumerate(target_actors):
                source_state = torch.load(
                    model_dir / f"actor_agent{agent_id}.pt", map_location=device
                )
                inflate_vector_state_dict(
                    actor, source_state, source_shape, target_shape, history_len
                )
            critic_state = torch.load(model_dir / "critic_agent.pt", map_location=device)
            inflate_vector_state_dict(
                target_critic, critic_state, source_shape, target_shape, history_len
            )
        else:
            for agent_id, actor in enumerate(target_actors):
                source_state = torch.load(
                    model_dir / f"actor_agent{agent_id}.pt", map_location=device
                )
                copy_compatible_state_dict(actor, source_state)
            critic_state = torch.load(model_dir / "critic_agent.pt", map_location=device)
            copy_compatible_state_dict(target_critic, critic_state)

    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=args.lr, eps=source_model_args["opti_eps"])
        for actor in target_actors
    ]
    critic_optimizer = torch.optim.Adam(
        target_critic.parameters(), lr=args.critic_lr, eps=source_model_args["opti_eps"]
    )

    print(f"Loaded teacher from: {model_dir}")
    if args.student_model_dir:
        print(f"Loaded student init from: {args.student_model_dir}")
    print(f"Loaded config from: {config_path}")
    print(f"Source obs space: {source_obs_space}; target obs space: {target_obs_space}")
    print(f"Source board: {source_shape}; target board: {target_shape}")
    print(f"Target observation type: {target_observation_type}; device: {device}")

    episode_actor_losses = []
    episode_value_losses = []
    start_time = time.time()
    for episode_index in range(1, args.episodes + 1):
        obs, share_obs, available_actions = source_env.reset()
        room_shape = tuple(source_env.env.room_state.shape)
        offset = fixed_offset(room_shape, target_shape, args.padding_mode, rng)
        actor_rnn_states = np.zeros(
            (n_agents, source_model_args["recurrent_n"], source_model_args["hidden_sizes"][-1]),
            dtype=np.float32,
        )
        critic_rnn_state = np.zeros(
            (1, source_model_args["recurrent_n"], source_model_args["hidden_sizes"][-1]),
            dtype=np.float32,
        )
        masks = np.ones((n_agents, 1), dtype=np.float32)
        critic_mask = np.ones((1, 1), dtype=np.float32)
        data = {
            "target_obs": [[] for _ in range(n_agents)],
            "available_actions": [[] for _ in range(n_agents)],
            "teacher_probs": [[] for _ in range(n_agents)],
            "target_share_obs": [],
            "teacher_values": [],
        }

        max_steps = args.max_episode_steps or int(getattr(source_env.env, "max_steps", 150))
        for _ in range(max_steps):
            actions, probs, value, actor_rnn_states, critic_rnn_state = teacher_step(
                source_actors,
                source_critic,
                obs,
                share_obs,
                available_actions,
                actor_rnn_states,
                critic_rnn_state,
                masks,
                critic_mask,
                args.rollout_action_mode,
                device,
            )
            for agent_id in range(n_agents):
                data["target_obs"][agent_id].append(
                    build_target_obs(
                        obs[agent_id],
                        source_observation_type,
                        target_observation_type,
                        room_shape,
                        target_shape,
                        offset,
                        action_history_len,
                        n_agents,
                        action_dim,
                    )
                )
                data["available_actions"][agent_id].append(
                    np.asarray(available_actions[agent_id], dtype=np.float32)
                )
                data["teacher_probs"][agent_id].append(probs[agent_id])
            data["target_share_obs"].append(
                build_target_obs(
                    share_obs[0],
                    source_observation_type,
                    target_observation_type,
                    room_shape,
                    target_shape,
                    offset,
                    action_history_len,
                    n_agents,
                    action_dim,
                )
            )
            data["teacher_values"].append(value)

            obs, share_obs, _, dones, _, available_actions = source_env.step(actions)
            if all(dones):
                break

        for _ in range(args.batch_epochs):
            actor_loss, value_loss = train_on_episode(
                target_actors,
                target_critic,
                actor_optimizers,
                critic_optimizer,
                data,
                target_model_args,
                args.temperature,
                args.value_loss_coef,
                device,
            )
        episode_actor_losses.append(actor_loss)
        episode_value_losses.append(value_loss)

        if episode_index % args.log_interval == 0 or episode_index == 1:
            elapsed = time.time() - start_time
            print(
                "episode "
                f"{episode_index}/{args.episodes} "
                f"actor_kl={np.mean(episode_actor_losses[-args.log_interval:]):.6f} "
                f"value_mse={np.mean(episode_value_losses[-args.log_interval:]):.6f} "
                f"steps={sum(len(item) for item in data['target_obs']) // n_agents} "
                f"offset={offset} elapsed={elapsed:.1f}s"
            )

    models_dir = save_models(
        args.output_dir,
        target_actors,
        target_critic,
        model_dir,
        run_config,
        args,
        args.copy_value_normalizer,
    )
    source_env.close()
    target_env.close()
    print(f"Saved distilled HARL models to: {models_dir}")


if __name__ == "__main__":
    main()
