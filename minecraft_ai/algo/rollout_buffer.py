"""On-policy rollout buffer for PPO.

Stores one rollout (128 steps) of experience, then provides
minibatch iteration for the PPO update.
"""

import torch
import numpy as np
from typing import Generator


class RolloutBuffer:
    """Stores transitions from a single rollout and yields minibatches."""

    def __init__(self, rollout_length: int, obs_shape: tuple,
                 n_action_dims: int, device: torch.device):
        self.rollout_length = rollout_length
        self.device = device
        self.pos = 0

        # Pre-allocate tensors
        self.observations = torch.zeros(
            (rollout_length, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (rollout_length, n_action_dims), dtype=torch.long, device=device
        )
        self.log_probs = torch.zeros(rollout_length, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(rollout_length, dtype=torch.float32, device=device)
        self.values = torch.zeros(rollout_length, dtype=torch.float32, device=device)
        self.dones = torch.zeros(rollout_length, dtype=torch.float32, device=device)

        # Computed after rollout
        self.advantages = None
        self.returns = None

    def add(self, obs, action, log_prob, reward, value, done):
        """Add a single transition."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.pos += 1

    def is_full(self) -> bool:
        return self.pos >= self.rollout_length

    def set_advantages(self, advantages: torch.Tensor, returns: torch.Tensor):
        """Set computed GAE advantages and returns."""
        self.advantages = advantages
        self.returns = returns

    def minibatch_iterator(self, num_minibatches: int) -> Generator:
        """Yield random minibatches for PPO update.

        Args:
            num_minibatches: number of minibatches to split data into

        Yields:
            dict with obs, actions, old_log_probs, advantages, returns, old_values
        """
        assert self.advantages is not None, "Call set_advantages before iterating"

        indices = np.arange(self.rollout_length)
        np.random.shuffle(indices)
        batch_size = self.rollout_length // num_minibatches

        for start in range(0, self.rollout_length, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            yield {
                "obs": self.observations[mb_indices],
                "actions": self.actions[mb_indices],
                "old_log_probs": self.log_probs[mb_indices],
                "advantages": self.advantages[mb_indices],
                "returns": self.returns[mb_indices],
                "old_values": self.values[mb_indices],
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.advantages = None
        self.returns = None
