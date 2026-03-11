"""Tests for Generalized Advantage Estimation."""

import torch
import pytest
from minecraft_ai.algo.gae import compute_gae


def test_basic_gae():
    """Hand-computed GAE for a simple 3-step trajectory."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    dones = torch.tensor([0.0, 0.0, 0.0])
    next_value = torch.tensor(2.0)
    gamma = 0.99
    lam = 0.95

    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma, lam)

    # delta_2 = 3.0 + 0.99 * 2.0 - 1.5 = 3.48
    # delta_1 = 2.0 + 0.99 * 1.5 - 1.0 = 2.485
    # delta_0 = 1.0 + 0.99 * 1.0 - 0.5 = 1.49
    # A_2 = delta_2 = 3.48
    # A_1 = delta_1 + 0.99 * 0.95 * A_2 = 2.485 + 0.9405 * 3.48 = 5.75694
    # A_0 = delta_0 + 0.99 * 0.95 * A_1 = 1.49 + 0.9405 * 5.75694 = 6.9042...

    assert advantages.shape == (3,)
    assert returns.shape == (3,)

    # Check delta_2
    assert abs(advantages[2].item() - 3.48) < 1e-4

    # Returns = advantages + values
    torch.testing.assert_close(returns, advantages + values)


def test_gae_with_done():
    """Done flag should cut the bootstrap."""
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    dones = torch.tensor([0.0, 1.0])  # episode ends at step 1
    next_value = torch.tensor(999.0)  # should be ignored
    gamma = 0.99
    lam = 0.95

    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma, lam)

    # delta_1 = 2.0 + 0.99 * 999.0 * 0.0 - 1.0 = 1.0  (done blocks bootstrap)
    # delta_0 = 1.0 + 0.99 * 1.0 - 0.5 = 1.49
    # A_1 = 1.0
    # A_0 = 1.49 + 0.99 * 0.95 * 0.0 * A_1 ... wait, done is at step 1
    # Actually: not_done[1] = 0.0, so A_1 = delta_1 = 1.0
    # not_done[0] = 1.0, so A_0 = delta_0 + gamma*lam*1.0*A_1 = 1.49 + 0.9405*1.0 = 2.4305

    assert abs(advantages[1].item() - 1.0) < 1e-4
    assert abs(advantages[0].item() - 2.4305) < 1e-3


def test_gae_all_zeros():
    """Zero rewards and values should give zero advantages."""
    rewards = torch.zeros(10)
    values = torch.zeros(10)
    dones = torch.zeros(10)
    next_value = torch.tensor(0.0)

    advantages, returns = compute_gae(rewards, values, dones, next_value)

    torch.testing.assert_close(advantages, torch.zeros(10))
    torch.testing.assert_close(returns, torch.zeros(10))
