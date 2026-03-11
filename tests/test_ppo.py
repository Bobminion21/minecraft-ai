"""Tests for the PPO algorithm."""

import torch
import pytest
from minecraft_ai.envs.action_space import MinecraftActionSpace
from minecraft_ai.models.policy_network import ActorCritic
from minecraft_ai.algo.ppo import PPO
from minecraft_ai.algo.rollout_buffer import RolloutBuffer
from minecraft_ai.algo.gae import compute_gae
from minecraft_ai.utils.config import Config


@pytest.fixture
def setup():
    torch.manual_seed(0)
    config = Config(rollout_length=32, ppo_epochs=1, num_minibatches=4)
    action_space = MinecraftActionSpace()
    model = ActorCritic(action_space, in_channels=12)
    ppo = PPO(model, config)
    buffer = RolloutBuffer(32, (12, 64, 64), 10, torch.device("cpu"))
    return config, model, ppo, buffer


def test_ppo_update(setup):
    """PPO update should run and return loss metrics."""
    config, model, ppo, buffer = setup

    # Fill buffer with fake data
    for _ in range(32):
        obs = torch.randn(12, 64, 64)
        with torch.no_grad():
            actions, log_prob, value, _ = model.act(obs.unsqueeze(0))
        buffer.add(obs, actions.squeeze(0), log_prob.squeeze(0),
                   reward=1.0, value=value.squeeze(0), done=False)

    # Compute GAE
    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones,
        torch.tensor(0.0), config.gamma, config.gae_lambda,
    )
    buffer.set_advantages(advantages, returns)

    # Run update
    metrics = ppo.update(buffer)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy_loss" in metrics
    assert "clip_fraction" in metrics
    # Clip fraction should be finite and reasonable
    assert 0.0 <= metrics["clip_fraction"] <= 1.0


def test_lr_annealing(setup):
    config, model, ppo, buffer = setup
    initial_lr = config.learning_rate

    ppo.update_learning_rate(0.5)
    current_lr = ppo.optimizer.param_groups[0]["lr"]
    assert abs(current_lr - initial_lr * 0.5) < 1e-8

    ppo.update_learning_rate(1.0)
    current_lr = ppo.optimizer.param_groups[0]["lr"]
    assert abs(current_lr) < 1e-8
