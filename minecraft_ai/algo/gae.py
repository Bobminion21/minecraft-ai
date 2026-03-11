"""Generalized Advantage Estimation (Schulman et al., 2016).

Computes advantages using the GAE(gamma, lambda) estimator:
  A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

This balances bias (low lambda) vs variance (high lambda) in advantage estimates.
"""

import torch
import numpy as np


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:
    """Compute GAE advantages and returns.

    Args:
        rewards: (T,) rewards at each step
        values: (T,) value estimates at each step
        dones: (T,) 1.0 if episode ended at this step, 0.0 otherwise
        next_value: scalar, V(s_T) bootstrap value
        gamma: discount factor
        gae_lambda: GAE lambda for bias-variance tradeoff

    Returns:
        advantages: (T,) GAE advantage estimates
        returns: (T,) discounted returns (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # If episode ended at step t, don't bootstrap from next state
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
