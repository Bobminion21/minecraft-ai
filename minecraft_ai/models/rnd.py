"""Random Network Distillation (Burda et al., 2019).

Provides intrinsic curiosity reward: a fixed random target network and a
predictor network. The prediction error on novel observations serves as
an exploration bonus -- high error = novel state = high intrinsic reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_utils import init_weights


class RNDNetwork(nn.Module):
    """Small CNN that maps observations to a feature vector."""

    def __init__(self, in_channels: int = 12, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """Random Network Distillation module.

    - target: fixed random network (never updated)
    - predictor: trained to match target output

    Intrinsic reward = MSE(predictor(obs), target(obs))
    """

    def __init__(self, in_channels: int = 12, feature_dim: int = 128,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.target = RNDNetwork(in_channels, feature_dim)
        self.predictor = RNDNetwork(in_channels, feature_dim)

        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

        # Initialize
        self.target.apply(lambda m: init_weights(m, gain=1.0))
        self.predictor.apply(lambda m: init_weights(m, gain=1.0))

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=learning_rate
        )

        # Running stats for normalizing intrinsic reward
        self.reward_running_mean = 0.0
        self.reward_running_var = 1.0
        self.reward_count = 0

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward for a batch of observations.

        Args:
            obs: (B, C, H, W) normalized observations

        Returns:
            (B,) intrinsic rewards (normalized)
        """
        with torch.no_grad():
            target_features = self.target(obs)
        predictor_features = self.predictor(obs)
        intrinsic_reward = (target_features - predictor_features).pow(2).mean(dim=-1)

        # Normalize
        normalized = self._normalize_reward(intrinsic_reward.detach())
        return normalized

    def update(self, obs: torch.Tensor) -> float:
        """Train predictor to match target.

        Args:
            obs: (B, C, H, W) observations

        Returns:
            prediction loss value
        """
        with torch.no_grad():
            target_features = self.target(obs)
        predictor_features = self.predictor(obs)

        loss = F.mse_loss(predictor_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Running normalization of intrinsic rewards."""
        batch_mean = reward.mean().item()
        batch_var = reward.var().item()
        batch_count = len(reward)

        # Welford's online algorithm
        delta = batch_mean - self.reward_running_mean
        total_count = self.reward_count + batch_count
        new_mean = self.reward_running_mean + delta * batch_count / max(total_count, 1)
        m_a = self.reward_running_var * self.reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / max(total_count, 1)
        new_var = m2 / max(total_count, 1)

        self.reward_running_mean = new_mean
        self.reward_running_var = new_var
        self.reward_count = total_count

        return (reward - self.reward_running_mean) / (self.reward_running_var ** 0.5 + 1e-8)
