"""Tests for checkpoint save/load."""

import torch
import pytest
import tempfile
from pathlib import Path
from minecraft_ai.envs.action_space import MinecraftActionSpace
from minecraft_ai.models.policy_network import ActorCritic
from minecraft_ai.training.checkpoint import CheckpointManager


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_save_and_load(tmpdir):
    action_space = MinecraftActionSpace()
    model = ActorCritic(action_space, in_channels=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mgr = CheckpointManager(tmpdir)

    # Save
    mgr.save(model, optimizer, step=1000, episode=10, curriculum_stage=1)

    # Create fresh model and load
    model2 = ActorCritic(action_space, in_channels=12)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    info = mgr.load(model2, optimizer2)

    assert info["step"] == 1000
    assert info["episode"] == 10
    assert info["curriculum_stage"] == 1

    # Weights should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_latest_exists(tmpdir):
    mgr = CheckpointManager(tmpdir)
    assert not mgr.latest_exists()

    action_space = MinecraftActionSpace()
    model = ActorCritic(action_space, in_channels=12)
    optimizer = torch.optim.Adam(model.parameters())
    mgr.save(model, optimizer, step=100, episode=1)

    assert mgr.latest_exists()


def test_cleanup(tmpdir):
    mgr = CheckpointManager(tmpdir, max_keep=2)
    action_space = MinecraftActionSpace()
    model = ActorCritic(action_space, in_channels=12)
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(5):
        mgr.save(model, optimizer, step=i * 100, episode=i)

    # Should keep only 2 numbered checkpoints + latest
    numbered = list(Path(tmpdir).glob("checkpoint_[0-9]*.pt"))
    assert len(numbered) <= 2
