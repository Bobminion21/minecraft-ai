"""Tests for the rollout buffer."""

import torch
import pytest
from minecraft_ai.algo.rollout_buffer import RolloutBuffer


@pytest.fixture
def buffer():
    return RolloutBuffer(
        rollout_length=8,
        obs_shape=(12, 64, 64),
        n_action_dims=10,
        device=torch.device("cpu"),
    )


def test_add_and_full(buffer):
    for i in range(8):
        buffer.add(
            obs=torch.randn(12, 64, 64),
            action=torch.randint(0, 2, (10,)),
            log_prob=torch.tensor(-1.0),
            reward=1.0,
            value=torch.tensor(0.5),
            done=False,
        )
    assert buffer.is_full()


def test_minibatch_iterator(buffer):
    for i in range(8):
        buffer.add(
            obs=torch.randn(12, 64, 64),
            action=torch.randint(0, 2, (10,)),
            log_prob=torch.tensor(-1.0),
            reward=1.0,
            value=torch.tensor(0.5),
            done=False,
        )
    buffer.set_advantages(torch.randn(8), torch.randn(8))

    batches = list(buffer.minibatch_iterator(num_minibatches=4))
    assert len(batches) == 4
    assert batches[0]["obs"].shape == (2, 12, 64, 64)
    assert batches[0]["actions"].shape == (2, 10)


def test_reset(buffer):
    buffer.add(
        obs=torch.randn(12, 64, 64),
        action=torch.randint(0, 2, (10,)),
        log_prob=torch.tensor(-1.0),
        reward=1.0,
        value=torch.tensor(0.5),
        done=False,
    )
    buffer.reset()
    assert buffer.pos == 0
    assert not buffer.is_full()
