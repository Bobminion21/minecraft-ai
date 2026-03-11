"""Tests for the mock Minecraft environment."""

import numpy as np
import pytest
from minecraft_ai.envs.mock_env import MockMinecraftEnv


@pytest.fixture
def env():
    e = MockMinecraftEnv(frame_size=64, max_steps=100)
    yield e
    e.close()


def test_reset(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)


def test_step(env):
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_truncation(env):
    """Episode should truncate at max_steps."""
    env.reset(seed=42)
    noop = env.action_space_handler.noop()
    for i in range(100):
        obs, reward, terminated, truncated, info = env.step(noop)
        if terminated:
            break
    # Should reach truncation at step 100 (if not terminated earlier)
    if not terminated:
        assert truncated


def test_reward_forward_attack():
    """Forward + attack should give positive reward on average."""
    env = MockMinecraftEnv(frame_size=64, max_steps=500)
    env.reset(seed=42)

    # Action with forward=1, attack=1
    action = env.action_space_handler.noop()
    action[0] = 1  # forward
    action[7] = 1  # attack

    rewards = []
    for _ in range(200):
        _, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        if term or trunc:
            env.reset()

    assert np.mean(rewards) > 0.05  # Should be positive on average
    env.close()
