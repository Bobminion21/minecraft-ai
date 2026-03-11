"""Proximal Policy Optimization (Schulman et al., 2017).

Implements:
- Clipped surrogate objective (prevents large policy updates)
- Clipped value loss (prevents value function from changing too fast)
- Entropy bonus (encourages exploration)
- Advantage normalization (stabilizes training)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from ..models.policy_network import ActorCritic
from .rollout_buffer import RolloutBuffer
from ..utils.config import Config


class PPO:
    """PPO trainer -- performs the actual gradient updates."""

    def __init__(self, model: ActorCritic, config: Config):
        self.model = model
        self.config = config

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )

        self.clip_epsilon = config.clip_epsilon
        self.value_clip = config.value_clip
        self.value_loss_coeff = config.value_loss_coeff
        self.entropy_coeff = config.entropy_coeff
        self.max_grad_norm = config.max_grad_norm

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update on a filled rollout buffer.

        Args:
            buffer: RolloutBuffer with advantages already computed

        Returns:
            dict of loss metrics for logging
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.minibatch_iterator(self.config.num_minibatches):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # Normalize advantages (per-minibatch)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Evaluate actions under current policy
                new_log_probs, new_values, entropy = self.model.evaluate_actions(obs, actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_pred_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.value_clip, self.value_clip
                )
                value_loss_unclipped = (new_values - returns) ** 2
                value_loss_clipped = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + self.value_loss_coeff * value_loss
                        + self.entropy_coeff * entropy_loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Tracking metrics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    approx_kl = ((ratio - 1.0) - (ratio.log())).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_clip_fraction += clip_fraction.item()
                total_approx_kl += approx_kl.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "total_loss": total_loss / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "approx_kl": total_approx_kl / num_updates,
        }

    def update_learning_rate(self, progress: float):
        """Linearly anneal LR from initial value to 0.

        Args:
            progress: fraction of training completed (0.0 to 1.0)
        """
        if self.config.anneal_lr:
            lr = self.config.learning_rate * (1.0 - progress)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
