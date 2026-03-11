"""Tests for Random Network Distillation."""

import torch
import pytest
from minecraft_ai.models.rnd import RND


def test_intrinsic_reward_shape():
    rnd = RND(in_channels=12, feature_dim=128)
    obs = torch.randn(4, 12, 64, 64)
    reward = rnd.compute_intrinsic_reward(obs)
    assert reward.shape == (4,)


def test_predictor_learns():
    """Intrinsic reward should decrease as predictor trains on repeated obs."""
    rnd = RND(in_channels=12, feature_dim=128)
    obs = torch.randn(8, 12, 64, 64)

    # Initial loss
    loss_before = rnd.update(obs)

    # Train for a bit on same observations
    for _ in range(50):
        loss = rnd.update(obs)

    # Loss should decrease
    assert loss < loss_before * 0.5, f"Loss didn't decrease: {loss_before:.4f} -> {loss:.4f}"
