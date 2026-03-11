"""Actor-Critic network with multi-discrete action heads.

Architecture:
  Shared encoder (IMPALA CNN) -> 256-dim features
    -> Policy heads: one Categorical per action dimension (10 heads)
    -> Value head: Linear -> scalar

The factored policy means each action dimension is independent,
which is much more parameter-efficient than a single giant Categorical.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

from .cnn_encoder import IMPALAEncoder
from ..utils.torch_utils import init_weights
from ..envs.action_space import MinecraftActionSpace


class ActorCritic(nn.Module):
    """PPO Actor-Critic with factored multi-discrete policy."""

    def __init__(self, action_space: MinecraftActionSpace,
                 in_channels: int = 12, feature_dim: int = 256):
        super().__init__()
        self.action_space = action_space
        self.nvec = action_space.nvec

        self.encoder = IMPALAEncoder(in_channels, feature_dim)

        # One linear head per action dimension
        self.policy_heads = nn.ModuleList([
            nn.Linear(feature_dim, int(n)) for n in self.nvec
        ])

        # Value head
        self.value_head = nn.Linear(feature_dim, 1)

        # Initialize policy heads with small weights, value head with unit gain
        for head in self.policy_heads:
            init_weights(head, gain=0.01)
        init_weights(self.value_head, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """Compute policy logits and value.

        Args:
            obs: (B, C, 64, 64) observation tensor

        Returns:
            logits: list of (B, n_i) tensors, one per action dimension
            value: (B,) value estimates
        """
        features = self.encoder(obs)
        logits = [head(features) for head in self.policy_heads]
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and return action, log_prob, value, entropy.

        Args:
            obs: (B, C, 64, 64)
            deterministic: if True, take argmax instead of sampling

        Returns:
            actions: (B, n_dims) int tensor
            log_prob: (B,) total log probability
            value: (B,) value estimate
            entropy: (B,) total entropy
        """
        logits, value = self.forward(obs)

        actions = []
        log_probs = []
        entropies = []

        for logit in logits:
            dist = Categorical(logits=logit)
            if deterministic:
                a = logit.argmax(dim=-1)
            else:
                a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

        actions = torch.stack(actions, dim=-1)        # (B, n_dims)
        log_prob = torch.stack(log_probs, dim=-1).sum(-1)  # (B,)
        entropy = torch.stack(entropies, dim=-1).sum(-1)   # (B,)

        return actions, log_prob, value, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions under current policy.

        Used during PPO update to compute ratios and losses.

        Args:
            obs: (B, C, 64, 64)
            actions: (B, n_dims) int tensor

        Returns:
            log_prob: (B,) total log probability of actions
            value: (B,) value estimate
            entropy: (B,) total entropy
        """
        logits, value = self.forward(obs)

        log_probs = []
        entropies = []

        for i, logit in enumerate(logits):
            dist = Categorical(logits=logit)
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        log_prob = torch.stack(log_probs, dim=-1).sum(-1)
        entropy = torch.stack(entropies, dim=-1).sum(-1)

        return log_prob, value, entropy
