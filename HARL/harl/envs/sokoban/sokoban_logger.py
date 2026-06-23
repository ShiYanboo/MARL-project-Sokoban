import numpy as np

from harl.common.base_logger import BaseLogger


class SokobanLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args["scenario"]

    def init(self, episodes):
        super().init(episodes)
        self._reset_train_trackers()
        self.train_finished_episode_stats = self._empty_metric_store("train")

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        super().episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer
        )
        self._flush_metric_store(self.train_finished_episode_stats)
        self.train_finished_episode_stats = self._empty_metric_store("train")

    def eval_init(self):
        super().eval_init()
        self._reset_eval_trackers()
        self.eval_finished_episode_stats = self._empty_metric_store("eval")

    def per_step(self, data):
        super().per_step(data)
        self._update_train_trackers(data)

    def eval_per_step(self, eval_data):
        super().eval_per_step(eval_data)
        self._update_eval_trackers(eval_data)

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self._finalize_eval_thread(tid)

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        self._flush_metric_store(self.eval_finished_episode_stats)
        self.eval_finished_episode_stats = self._empty_metric_store("eval")

    def _empty_metric_store(self, stage):
        prefix = f"sokoban/{stage}_"
        return {
            f"{prefix}episode_length": [],
            f"{prefix}mean_step_reward": [],
            f"{prefix}mean_base_reward": [],
            f"{prefix}mean_shaping_reward": [],
            f"{prefix}mean_distance_shaping_reward": [],
            f"{prefix}mean_pushability_shaping_reward": [],
            f"{prefix}mean_deadlock_shaping_reward": [],
            f"{prefix}mean_agent_box_distance_shaping_reward": [],
            f"{prefix}box_pushes": [],
            f"{prefix}player_moves": [],
            f"{prefix}noop_rate": [],
            f"{prefix}conflict_rate": [],
            f"{prefix}invalid_action_rate": [],
            f"{prefix}success_rate": [],
            f"{prefix}final_boxes_on_target": [],
            f"{prefix}box_completion_ratio": [],
            f"{prefix}agent0_execution_rate": [],
            f"{prefix}agent1_execution_rate": [],
        }

    def _reset_train_trackers(self):
        n_threads = self.algo_args["train"]["n_rollout_threads"]
        self.train_episode_lengths = np.zeros(n_threads, dtype=np.int32)
        self.train_step_reward_sums = np.zeros(n_threads, dtype=np.float32)
        self.train_reward_component_sums = self._empty_reward_component_sums(n_threads)
        self.train_box_pushes = np.zeros(n_threads, dtype=np.int32)
        self.train_player_moves = np.zeros(n_threads, dtype=np.int32)
        self.train_noops = np.zeros(n_threads, dtype=np.int32)
        self.train_conflicts = np.zeros(n_threads, dtype=np.int32)
        self.train_invalid_actions = np.zeros(n_threads, dtype=np.int32)
        self.train_agent0_exec = np.zeros(n_threads, dtype=np.int32)
        self.train_agent1_exec = np.zeros(n_threads, dtype=np.int32)
        self.train_last_boxes_on_target = np.zeros(n_threads, dtype=np.int32)
        self.train_last_num_boxes = np.ones(n_threads, dtype=np.int32)
        self.train_success = np.zeros(n_threads, dtype=np.float32)

    def _reset_eval_trackers(self):
        n_threads = self.algo_args["eval"]["n_eval_rollout_threads"]
        self.eval_episode_lengths = np.zeros(n_threads, dtype=np.int32)
        self.eval_step_reward_sums = np.zeros(n_threads, dtype=np.float32)
        self.eval_reward_component_sums = self._empty_reward_component_sums(n_threads)
        self.eval_box_pushes = np.zeros(n_threads, dtype=np.int32)
        self.eval_player_moves = np.zeros(n_threads, dtype=np.int32)
        self.eval_noops = np.zeros(n_threads, dtype=np.int32)
        self.eval_conflicts = np.zeros(n_threads, dtype=np.int32)
        self.eval_invalid_actions = np.zeros(n_threads, dtype=np.int32)
        self.eval_agent0_exec = np.zeros(n_threads, dtype=np.int32)
        self.eval_agent1_exec = np.zeros(n_threads, dtype=np.int32)
        self.eval_last_boxes_on_target = np.zeros(n_threads, dtype=np.int32)
        self.eval_last_num_boxes = np.ones(n_threads, dtype=np.int32)
        self.eval_success = np.zeros(n_threads, dtype=np.float32)

    def _update_train_trackers(self, data):
        _, _, rewards, dones, infos, _, _, _, _, _, _ = data
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()

        for thread_id, info_list in enumerate(infos):
            info = info_list[0]
            self.train_episode_lengths[thread_id] += 1
            self.train_step_reward_sums[thread_id] += reward_env[thread_id]
            self._update_reward_component_sums(
                self.train_reward_component_sums, thread_id, info
            )
            self.train_box_pushes[thread_id] += int(info.get("action_moved_box", False))
            self.train_player_moves[thread_id] += int(
                info.get("action_moved_player", False)
            )
            self.train_noops[thread_id] += int(info.get("noop_executed", False))
            self.train_conflicts[thread_id] += int(info.get("had_conflict", False))
            self.train_invalid_actions[thread_id] += int(
                info.get("invalid_action_attempt", False)
            )
            self.train_agent0_exec[thread_id] += int(info.get("chosen_agent", -1) == 0)
            self.train_agent1_exec[thread_id] += int(info.get("chosen_agent", -1) == 1)
            self.train_last_boxes_on_target[thread_id] = int(
                info.get("boxes_on_target", 0)
            )
            self.train_last_num_boxes[thread_id] = max(int(info.get("num_boxes", 1)), 1)
            self.train_success[thread_id] = float(info.get("success", False))

            if dones_env[thread_id]:
                self._finalize_train_thread(thread_id)

    def _update_eval_trackers(self, eval_data):
        _, _, eval_rewards, eval_dones, eval_infos, _ = eval_data
        eval_reward_env = np.mean(eval_rewards, axis=1).flatten()

        for thread_id, info_list in enumerate(eval_infos):
            info = info_list[0]
            self.eval_episode_lengths[thread_id] += 1
            self.eval_step_reward_sums[thread_id] += eval_reward_env[thread_id]
            self._update_reward_component_sums(
                self.eval_reward_component_sums, thread_id, info
            )
            self.eval_box_pushes[thread_id] += int(info.get("action_moved_box", False))
            self.eval_player_moves[thread_id] += int(
                info.get("action_moved_player", False)
            )
            self.eval_noops[thread_id] += int(info.get("noop_executed", False))
            self.eval_conflicts[thread_id] += int(info.get("had_conflict", False))
            self.eval_invalid_actions[thread_id] += int(
                info.get("invalid_action_attempt", False)
            )
            self.eval_agent0_exec[thread_id] += int(info.get("chosen_agent", -1) == 0)
            self.eval_agent1_exec[thread_id] += int(info.get("chosen_agent", -1) == 1)
            self.eval_last_boxes_on_target[thread_id] = int(
                info.get("boxes_on_target", 0)
            )
            self.eval_last_num_boxes[thread_id] = max(int(info.get("num_boxes", 1)), 1)
            self.eval_success[thread_id] = float(info.get("success", False))

    def _finalize_train_thread(self, thread_id):
        self._append_episode_stats(
            self.train_finished_episode_stats,
            self.train_episode_lengths[thread_id],
            self.train_step_reward_sums[thread_id],
            {
                name: values[thread_id]
                for name, values in self.train_reward_component_sums.items()
            },
            self.train_box_pushes[thread_id],
            self.train_player_moves[thread_id],
            self.train_noops[thread_id],
            self.train_conflicts[thread_id],
            self.train_invalid_actions[thread_id],
            self.train_agent0_exec[thread_id],
            self.train_agent1_exec[thread_id],
            self.train_last_boxes_on_target[thread_id],
            self.train_last_num_boxes[thread_id],
            self.train_success[thread_id],
        )
        self._reset_train_thread(thread_id)

    def _finalize_eval_thread(self, thread_id):
        self._append_episode_stats(
            self.eval_finished_episode_stats,
            self.eval_episode_lengths[thread_id],
            self.eval_step_reward_sums[thread_id],
            {
                name: values[thread_id]
                for name, values in self.eval_reward_component_sums.items()
            },
            self.eval_box_pushes[thread_id],
            self.eval_player_moves[thread_id],
            self.eval_noops[thread_id],
            self.eval_conflicts[thread_id],
            self.eval_invalid_actions[thread_id],
            self.eval_agent0_exec[thread_id],
            self.eval_agent1_exec[thread_id],
            self.eval_last_boxes_on_target[thread_id],
            self.eval_last_num_boxes[thread_id],
            self.eval_success[thread_id],
        )
        self._reset_eval_thread(thread_id)

    def _append_episode_stats(
        self,
        metric_store,
        episode_length,
        step_reward_sum,
        reward_component_sums,
        box_pushes,
        player_moves,
        noops,
        conflicts,
        invalid_actions,
        agent0_exec,
        agent1_exec,
        boxes_on_target,
        num_boxes,
        success,
    ):
        length = max(int(episode_length), 1)
        stage = "train" if "sokoban/train_episode_length" in metric_store else "eval"
        metric_store[f"sokoban/{stage}_episode_length"].append(length)
        metric_store[f"sokoban/{stage}_mean_step_reward"].append(
            float(step_reward_sum) / length
        )
        for name, value in reward_component_sums.items():
            metric_store[f"sokoban/{stage}_mean_{name}"].append(
                float(value) / length
            )
        metric_store[f"sokoban/{stage}_box_pushes"].append(int(box_pushes))
        metric_store[f"sokoban/{stage}_player_moves"].append(int(player_moves))
        metric_store[f"sokoban/{stage}_noop_rate"].append(float(noops) / length)
        metric_store[f"sokoban/{stage}_conflict_rate"].append(float(conflicts) / length)
        metric_store[f"sokoban/{stage}_invalid_action_rate"].append(
            float(invalid_actions) / length
        )
        metric_store[f"sokoban/{stage}_success_rate"].append(float(success))
        metric_store[f"sokoban/{stage}_final_boxes_on_target"].append(
            int(boxes_on_target)
        )
        metric_store[f"sokoban/{stage}_box_completion_ratio"].append(
            float(boxes_on_target) / max(int(num_boxes), 1)
        )
        metric_store[f"sokoban/{stage}_agent0_execution_rate"].append(
            float(agent0_exec) / length
        )
        metric_store[f"sokoban/{stage}_agent1_execution_rate"].append(
            float(agent1_exec) / length
        )

    def _reset_train_thread(self, thread_id):
        self.train_episode_lengths[thread_id] = 0
        self.train_step_reward_sums[thread_id] = 0.0
        for values in self.train_reward_component_sums.values():
            values[thread_id] = 0.0
        self.train_box_pushes[thread_id] = 0
        self.train_player_moves[thread_id] = 0
        self.train_noops[thread_id] = 0
        self.train_conflicts[thread_id] = 0
        self.train_invalid_actions[thread_id] = 0
        self.train_agent0_exec[thread_id] = 0
        self.train_agent1_exec[thread_id] = 0
        self.train_last_boxes_on_target[thread_id] = 0
        self.train_last_num_boxes[thread_id] = 1
        self.train_success[thread_id] = 0.0

    def _reset_eval_thread(self, thread_id):
        self.eval_episode_lengths[thread_id] = 0
        self.eval_step_reward_sums[thread_id] = 0.0
        for values in self.eval_reward_component_sums.values():
            values[thread_id] = 0.0
        self.eval_box_pushes[thread_id] = 0
        self.eval_player_moves[thread_id] = 0
        self.eval_noops[thread_id] = 0
        self.eval_conflicts[thread_id] = 0
        self.eval_invalid_actions[thread_id] = 0
        self.eval_agent0_exec[thread_id] = 0
        self.eval_agent1_exec[thread_id] = 0
        self.eval_last_boxes_on_target[thread_id] = 0
        self.eval_last_num_boxes[thread_id] = 1
        self.eval_success[thread_id] = 0.0

    def _flush_metric_store(self, metric_store):
        for metric_name, values in metric_store.items():
            if values:
                self.writter.add_scalar(
                    metric_name, float(np.mean(values)), self.total_num_steps
                )

    @staticmethod
    def _empty_reward_component_sums(n_threads):
        return {
            "base_reward": np.zeros(n_threads, dtype=np.float32),
            "shaping_reward": np.zeros(n_threads, dtype=np.float32),
            "distance_shaping_reward": np.zeros(n_threads, dtype=np.float32),
            "pushability_shaping_reward": np.zeros(n_threads, dtype=np.float32),
            "deadlock_shaping_reward": np.zeros(n_threads, dtype=np.float32),
            "agent_box_distance_shaping_reward": np.zeros(
                n_threads, dtype=np.float32
            ),
        }

    @staticmethod
    def _update_reward_component_sums(component_sums, thread_id, info):
        for name, values in component_sums.items():
            values[thread_id] += float(info.get(name, 0.0))
