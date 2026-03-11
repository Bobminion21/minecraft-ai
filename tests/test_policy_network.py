"""Tests for the Actor-Critic policy network."""

import torch
import pytest
from minecraft_ai.envs.action_space import MinecraftActionSpace
from minecraft_ai.models.policy_network import ActorCritic


@pytest.fixture
def model():
    action_space = MinecraftActionSpace()
    return ActorCritic(action_space, in_channels=12, feature_dim=256)


def test_act(model):
    obs = torch.randn(4, 12, 64, 64)
    actions, log_prob, value, entropy = model.act(obs)
    assert actions.shape == (4, 10)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert entropy.shape == (4,)


def test_act_deterministic(model):
    obs = torch.randn(2, 12, 64, 64)
    a1, _, _, _ = model.act(obs, deterministic=True)
    a2, _, _, _ = model.act(obs, deterministic=True)
    torch.testing.assert_close(a1, a2)


def test_evaluate_actions(model):
    obs = torch.randn(4, 12, 64, 64)
    actions, old_lp, _, _ = model.act(obs)
    new_lp, value, entropy = model.evaluate_actions(obs, actions)
    assert new_lp.shape == (4,)
    assert value.shape == (4,)
    assert entropy.shape == (4,)
    # Log probs should match (same params, same actions)
    torch.testing.assert_close(old_lp, new_lp)


def test_action_bounds(model):
    """All sampled actions should be within the valid range."""
    obs = torch.randn(16, 12, 64, 64)
    actions, _, _, _ = model.act(obs)
    action_space = MinecraftActionSpace()
    for i, n in enumerate(action_space.nvec):
        assert (actions[:, i] >= 0).all()
        assert (actions[:, i] < n).all()
