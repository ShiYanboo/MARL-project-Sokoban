"""Runner for HAHyPO."""

import numpy as np

from harl.runners.on_policy_ha_runner import OnPolicyHARunner


class OnPolicyHAHyPORunner(OnPolicyHARunner):
    """HAPPO runner with hybrid group-relative actor advantages.

    The critic still learns the normal team return. Only the actor advantage is
    changed:

        A = (1 - alpha) * normalize(A_GAE) + alpha * A_group

    where A_group is a z-scored trajectory return across rollout threads in the
    current update batch and then broadcast to all steps of that trajectory.
    """

    def _advantages_with_actor_credit(self, advantages, agent_id):
        hybrid_advantages = self._hybrid_group_advantages(advantages)
        return super()._advantages_with_actor_credit(hybrid_advantages, agent_id)

    def _hybrid_group_advantages(self, advantages):
        alpha = float(self.algo_args["algo"].get("hybrid_alpha", 0.0) or 0.0)
        if alpha <= 0.0:
            return advantages.copy()
        if alpha > 1.0:
            raise ValueError("HAHyPO hybrid_alpha must be in [0, 1].")
        if self.state_type != "EP":
            raise NotImplementedError("HAHyPO group-relative advantage expects EP state.")

        gae_advantages = self._normalize_advantages(advantages)
        group_advantages = self._group_relative_advantages(advantages.shape)
        return (1.0 - alpha) * gae_advantages + alpha * group_advantages

    def _normalize_advantages(self, advantages):
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        if not np.isfinite(std_advantages) or std_advantages < 1e-8:
            return np.zeros_like(advantages)
        return (advantages - mean_advantages) / (std_advantages + 1e-5)

    def _group_relative_advantages(self, target_shape):
        rewards = self.critic_buffer.rewards.astype(np.float32)
        if rewards.ndim != 3:
            raise ValueError(
                "HAHyPO EP reward buffer should have shape (T, N, 1), "
                f"got {rewards.shape}."
            )

        if self.algo_args["algo"].get("hybrid_use_discounted_return", False):
            gamma = float(self.algo_args["algo"].get("gamma", 0.99))
            discounts = (gamma ** np.arange(rewards.shape[0], dtype=np.float32))[
                :, None, None
            ]
            trajectory_returns = np.sum(rewards * discounts, axis=0)
        else:
            trajectory_returns = np.sum(rewards, axis=0)

        mean_return = float(np.mean(trajectory_returns))
        std_return = float(np.std(trajectory_returns))
        eps = float(self.algo_args["algo"].get("hybrid_group_eps", 1e-5))
        if not np.isfinite(std_return) or std_return < eps:
            return np.zeros(target_shape, dtype=np.float32)

        group_advantages = (trajectory_returns - mean_return) / (std_return + eps)
        clip_value = self.algo_args["algo"].get("hybrid_group_clip", 5.0)
        if clip_value is not None:
            clip_value = float(clip_value)
            group_advantages = np.clip(group_advantages, -clip_value, clip_value)

        return np.broadcast_to(group_advantages[None, :, :], target_shape).astype(
            np.float32
        )
