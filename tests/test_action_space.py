"""Tests for action space conversion."""

import numpy as np
import pytest
from minecraft_ai.envs.action_space import (
    MinecraftActionSpace, CAMERA_BINS, NUM_BUTTONS, NUM_CAMERA_BINS,
)


@pytest.fixture
def action_space():
    return MinecraftActionSpace()


def test_space_shape(action_space):
    assert action_space.n_dims == 10  # 8 buttons + 2 camera
    assert len(action_space.nvec) == 10


def test_noop(action_space):
    noop = action_space.noop()
    assert noop.shape == (10,)
    # All buttons off
    assert all(noop[:NUM_BUTTONS] == 0)
    # Camera centered (index 5 = 0 degrees)
    assert noop[NUM_BUTTONS] == NUM_CAMERA_BINS // 2
    assert noop[NUM_BUTTONS + 1] == NUM_CAMERA_BINS // 2


def test_sample(action_space):
    action = action_space.sample()
    assert action.shape == (10,)
    # Check bounds
    for i, n in enumerate(action_space.nvec):
        assert 0 <= action[i] < n


def test_roundtrip(action_space):
    """MultiDiscrete -> MineRL dict -> MultiDiscrete should be identity."""
    for _ in range(20):
        original = action_space.sample()
        minerl = action_space.to_minerl(original)
        recovered = action_space.from_minerl(minerl)
        np.testing.assert_array_equal(original, recovered)


def test_to_minerl_format(action_space):
    action = np.array([1, 0, 0, 0, 1, 0, 0, 1, 3, 7], dtype=np.int64)
    minerl = action_space.to_minerl(action)
    assert minerl["forward"] == 1
    assert minerl["back"] == 0
    assert minerl["jump"] == 1
    assert minerl["attack"] == 1
    assert minerl["camera"].shape == (2,)
    assert minerl["camera"][0] == CAMERA_BINS[3]
    assert minerl["camera"][1] == CAMERA_BINS[7]
