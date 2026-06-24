import copy
import sys
from pathlib import Path

import gym
import numpy as np
from gym.spaces import Box, Discrete

from harl.envs.sokoban.reward_shaping import SokobanRewardShaper


def _ensure_local_gym_sokoban():
    try:
        import gym_sokoban  # noqa: F401
        return
    except ImportError:
        workspace_root = Path(__file__).resolve().parents[4]
        local_repo = workspace_root / "gym-sokoban"
        if not local_repo.exists():
            raise
        sys.path.insert(0, str(local_repo))
        import gym_sokoban  # noqa: F401


class SokobanEnv:
    def __init__(self, args):
        _ensure_local_gym_sokoban()

        self.args = copy.deepcopy(args)
        self.scenario = self.args.get("scenario", "TwoPlayer-Sokoban-v0")
        self.state_type = self.args.get("state_type", "EP")
        self.control_mode = self.args.get("control_mode", "turn_based")
        self.observation_type = self.args.get("observation_type", "vector")
        self.conflict_resolution = self.args.get("conflict_resolution", "round_robin")
        self.max_steps = self.args.get("max_steps")
        self.reward_scale = float(self.args.get("reward_scale", 1.0))
        self.use_reward_shaping = bool(
            self.args.get("use_reward_shaping", False)
        )
        self.reward_shaper = SokobanRewardShaper(
            distance_weight=(
                self.args.get("distance_shaping_weight", 0.05)
                if self.use_reward_shaping
                else 0.0
            ),
            pushability_weight=(
                self.args.get("pushability_shaping_weight", 0.02)
                if self.use_reward_shaping
                else 0.0
            ),
            deadlock_penalty=(
                self.args.get("deadlock_penalty", 2.0)
                if self.use_reward_shaping
                else 0.0
            ),
            deadlock_penalty_mode=self.args.get("deadlock_penalty_mode", "state"),
            agent_box_distance_weight=(
                self.args.get("agent_box_distance_shaping_weight", 0.005)
                if self.use_reward_shaping
                else 0.0
            ),
            useful_push_weight=(
                self.args.get("useful_push_shaping_weight", 0.0)
                if self.use_reward_shaping
                else 0.0
            ),
        )
        if self.control_mode not in {"turn_based", "joint_resolve"}:
            raise ValueError(
                f"Unsupported Sokoban control_mode: {self.control_mode}. "
                "Choose from: turn_based, joint_resolve."
            )
        if self.observation_type not in {"vector", "cnn"}:
            raise ValueError(
                f"Unsupported Sokoban observation_type: {self.observation_type}. "
                "Choose from: vector, cnn."
            )

        make_kwargs = {}
        for key in ["dim_room", "max_steps", "num_boxes", "num_gen_steps"]:
            value = self.args.get(key)
            if value is not None:
                if key == "dim_room" and isinstance(value, int):
                    value = (value, value)
                make_kwargs[key] = value

        self.env = gym.make(self.scenario, **make_kwargs)
        if self.args.get("reward_finished") is not None:
            self.env.unwrapped.reward_finished = float(
                self.args["reward_finished"]
            )
        self.n_agents = 2
        self._seed = 0
        self._priority_agent = 0
        self._rng = np.random.default_rng()
        self._reset_retry_limit = int(self.args.get("reset_retry_limit", 20))

        self._reset_env()

        obs_shape = self._build_agent_obs(agent_id=0).shape
        state_shape = self._build_shared_state().shape
        self.observation_space = [
            Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=0.0, high=1.0, shape=state_shape, dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [Discrete(9) for _ in range(self.n_agents)]

    def step(self, actions):
        actions = np.asarray(actions).reshape(self.n_agents, -1)
        discrete_actions = [int(np.asarray(action).item()) for action in actions]
        chosen_agent, env_action, resolution_info = self._resolve_action(discrete_actions)

        before_state = (
            self.reward_shaper.snapshot(self.env)
            if self.reward_shaper.enabled
            else None
        )
        _, base_reward, done, env_info = self.env.step(env_action)
        base_reward = float(base_reward) * self.reward_scale
        if self.reward_shaper.enabled:
            after_state = self.reward_shaper.snapshot(self.env)
            shaping = self.reward_shaper.evaluate_transition(before_state, after_state)
        else:
            shaping = self.reward_shaper.empty_result()
        reward = base_reward + shaping["total"]
        self._priority_agent = 1 - self._priority_agent

        obs = self._get_obs()
        share_obs = self._get_share_obs()

        info = {
            "chosen_agent": chosen_agent,
            "env_action": env_action,
            "priority_agent_next": self._priority_agent,
            "control_mode": self.control_mode,
            "joint_actions": discrete_actions,
            "active_agents": resolution_info["active_agents"],
            "had_conflict": resolution_info["had_conflict"],
            "noop_executed": resolution_info["noop_executed"],
            "invalid_action_attempt": resolution_info["invalid_action_attempt"],
            "boxes_on_target": int(getattr(self.env, "boxes_on_target", 0)),
            "num_boxes": int(getattr(self.env, "num_boxes", 0)),
            "box_completion_ratio": float(
                getattr(self.env, "boxes_on_target", 0)
                / max(getattr(self.env, "num_boxes", 1), 1)
            ),
            "steps_used": int(getattr(self.env, "num_env_steps", 0)),
            "max_steps": int(getattr(self.env, "max_steps", 0)),
            "step_reward": reward,
            "base_reward": base_reward,
            "shaping_reward": shaping["total"],
            "distance_shaping_reward": shaping["distance"],
            "pushability_shaping_reward": shaping["pushability"],
            "deadlock_shaping_reward": shaping["deadlock"],
            "agent_box_distance_shaping_reward": shaping["agent_box_distance"],
            "useful_push_shaping_reward": shaping["useful_push"],
            "box_target_distance": shaping["box_target_distance_after"],
            "pushability": shaping["pushability_after"],
            "deadlocked_boxes": shaping["deadlocked_boxes"],
            "deadlock_penalty_count": shaping["deadlock_penalty_count"],
            "agent_box_distance": shaping["agent_box_distance_after"],
            "useful_push_applied": shaping["useful_push_applied"],
            "useful_push_distance_delta": shaping["useful_push_distance_delta"],
            "action_moved_player": bool(env_info.get("action.moved_player", False)),
            "action_moved_box": bool(env_info.get("action.moved_box", False)),
            **env_info,
        }
        info["success"] = bool(done and env_info.get("all_boxes_on_target", False))
        if done and env_info.get("maxsteps_used", False):
            info["bad_transition"] = True

        rewards = [[reward] for _ in range(self.n_agents)]
        dones = [done for _ in range(self.n_agents)]
        infos = [copy.deepcopy(info) for _ in range(self.n_agents)]
        return obs, share_obs, rewards, dones, infos, self.get_avail_actions()

    def reset(self):
        self.env.seed(self._seed)
        self._seed += 1
        self._priority_agent = 0
        self._reset_env()
        return self._get_obs(), self._get_share_obs(), self.get_avail_actions()

    def get_avail_actions(self):
        if self.control_mode == "turn_based":
            avail_actions = []
            for agent_id in range(self.n_agents):
                agent_avail = [0] * self.action_space[agent_id].n
                if agent_id == self._priority_agent:
                    agent_avail = [1] * self.action_space[agent_id].n
                else:
                    agent_avail[0] = 1
                avail_actions.append(agent_avail)
            return avail_actions
        return [[1] * self.action_space[agent_id].n for agent_id in range(self.n_agents)]

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def _resolve_action(self, actions):
        if self.control_mode == "turn_based":
            chosen_agent = self._priority_agent
            chosen_action = actions[chosen_agent]
            invalid_action_attempt = any(
                actions[agent_id] != 0
                for agent_id in range(self.n_agents)
                if agent_id != chosen_agent
            )
            resolution_info = {
                "active_agents": [chosen_agent] if chosen_action != 0 else [],
                "had_conflict": False,
                "noop_executed": chosen_action == 0,
                "invalid_action_attempt": invalid_action_attempt,
            }
            if chosen_action == 0:
                return chosen_agent, 0, resolution_info
            if chosen_agent == 0:
                return chosen_agent, chosen_action, resolution_info
            return chosen_agent, chosen_action + 8, resolution_info

        active_agents = [agent_id for agent_id, action in enumerate(actions) if action != 0]
        resolution_info = {
            "active_agents": active_agents,
            "had_conflict": len(active_agents) > 1,
            "noop_executed": len(active_agents) == 0,
            "invalid_action_attempt": False,
        }
        if not active_agents:
            return -1, 0, resolution_info

        if len(active_agents) == 1:
            chosen_agent = active_agents[0]
        elif self.conflict_resolution == "random":
            chosen_agent = int(self._rng.choice(active_agents))
        else:
            chosen_agent = self._priority_agent

        action = actions[chosen_agent]
        if chosen_agent == 0:
            return chosen_agent, action, resolution_info
        return chosen_agent, action + 8, resolution_info

    def _reset_env(self):
        last_error = None
        for _ in range(self._reset_retry_limit):
            try:
                self.env.reset()
                if len(np.argwhere(self.env.room_state == 5)) >= 2:
                    return
            except IndexError as error:
                last_error = error
        raise RuntimeError(
            f"Failed to generate a valid two-player Sokoban room for {self.scenario}."
        ) from last_error

    def _get_obs(self):
        return [self._build_agent_obs(agent_id) for agent_id in range(self.n_agents)]

    def _get_share_obs(self):
        state = self._build_shared_state()
        return [state.copy() for _ in range(self.n_agents)]

    def _build_agent_obs(self, agent_id):
        room_fixed = self.env.room_fixed
        room_state = self.env.room_state
        self_pos = self.env.player_positions[agent_id]
        other_pos = self.env.player_positions[1 - agent_id]
        channels = [
            (room_fixed == 0).astype(np.float32),
            (room_fixed == 2).astype(np.float32),
            (room_state == 3).astype(np.float32),
            (room_state == 4).astype(np.float32),
            self._position_map(self_pos, room_state.shape),
            self._position_map(other_pos, room_state.shape),
        ]
        if self.observation_type == "cnn":
            channels.extend(
                [
                    self._constant_map(
                        float(self._priority_agent == agent_id), room_state.shape
                    ),
                    self._constant_map(
                        float(self._priority_agent != agent_id), room_state.shape
                    ),
                    self._constant_map(
                        self.env.num_env_steps / max(self.env.max_steps, 1),
                        room_state.shape,
                    ),
                ]
            )
            return np.stack(channels, axis=0).astype(np.float32)

        flat = np.concatenate([channel.reshape(-1) for channel in channels], axis=0)
        extras = np.array(
            [
                float(self._priority_agent == 0),
                float(self._priority_agent == 1),
                self.env.num_env_steps / max(self.env.max_steps, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([flat, extras], axis=0).astype(np.float32)

    def _build_shared_state(self):
        room_fixed = self.env.room_fixed
        room_state = self.env.room_state
        channels = [
            (room_fixed == 0).astype(np.float32),
            (room_fixed == 2).astype(np.float32),
            (room_state == 3).astype(np.float32),
            (room_state == 4).astype(np.float32),
            self._position_map(self.env.player_positions[0], room_state.shape),
            self._position_map(self.env.player_positions[1], room_state.shape),
        ]
        if self.observation_type == "cnn":
            channels.extend(
                [
                    self._constant_map(
                        float(self._priority_agent == 0), room_state.shape
                    ),
                    self._constant_map(
                        float(self._priority_agent == 1), room_state.shape
                    ),
                    self._constant_map(
                        self.env.num_env_steps / max(self.env.max_steps, 1),
                        room_state.shape,
                    ),
                ]
            )
            return np.stack(channels, axis=0).astype(np.float32)

        flat = np.concatenate([channel.reshape(-1) for channel in channels], axis=0)
        extras = np.array(
            [
                float(self._priority_agent == 0),
                float(self._priority_agent == 1),
                self.env.num_env_steps / max(self.env.max_steps, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([flat, extras], axis=0).astype(np.float32)

    @staticmethod
    def _position_map(position, shape):
        arr = np.zeros(shape, dtype=np.float32)
        arr[tuple(position)] = 1.0
        return arr

    @staticmethod
    def _constant_map(value, shape):
        return np.full(shape, value, dtype=np.float32)
