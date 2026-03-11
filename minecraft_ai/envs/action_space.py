"""Discretize MineRL's Dict action space into MultiDiscrete.

MineRL actions are a dict with continuous camera + binary buttons.
We convert to independent categorical distributions:
  - 8 binary buttons (forward, back, left, right, jump, sneak, sprint, attack)
  - 2 camera axes (pitch, yaw) each discretized into 11 bins
Total: 10 action dimensions.
"""

import numpy as np
from gymnasium import spaces


# Camera angle bins: [-10, -8, ..., -2, 0, 2, ..., 8, 10] degrees
CAMERA_BINS = np.linspace(-10, 10, 11)
NUM_CAMERA_BINS = len(CAMERA_BINS)

BUTTON_KEYS = [
    "forward", "back", "left", "right",
    "jump", "sneak", "sprint", "attack",
]
NUM_BUTTONS = len(BUTTON_KEYS)


class MinecraftActionSpace:
    """Converts between MultiDiscrete actions and MineRL dict actions."""

    def __init__(self):
        # 8 binary buttons + 2 camera axes (11 bins each)
        self.nvec = np.array(
            [2] * NUM_BUTTONS + [NUM_CAMERA_BINS, NUM_CAMERA_BINS],
            dtype=np.int64,
        )
        self.space = spaces.MultiDiscrete(self.nvec)
        self.n_dims = len(self.nvec)

    def to_minerl(self, action: np.ndarray) -> dict:
        """Convert MultiDiscrete action array to MineRL dict action."""
        minerl_action = {
            key: int(action[i]) for i, key in enumerate(BUTTON_KEYS)
        }
        pitch = float(CAMERA_BINS[action[NUM_BUTTONS]])
        yaw = float(CAMERA_BINS[action[NUM_BUTTONS + 1]])
        minerl_action["camera"] = np.array([pitch, yaw], dtype=np.float32)
        return minerl_action

    def from_minerl(self, minerl_action: dict) -> np.ndarray:
        """Convert MineRL dict action to MultiDiscrete array."""
        action = np.zeros(self.n_dims, dtype=np.int64)
        for i, key in enumerate(BUTTON_KEYS):
            action[i] = int(minerl_action.get(key, 0))
        camera = minerl_action.get("camera", np.zeros(2))
        action[NUM_BUTTONS] = int(np.argmin(np.abs(CAMERA_BINS - camera[0])))
        action[NUM_BUTTONS + 1] = int(np.argmin(np.abs(CAMERA_BINS - camera[1])))
        return action

    def sample(self) -> np.ndarray:
        """Random action."""
        return self.space.sample()

    def noop(self) -> np.ndarray:
        """No-op action (all zeros, camera centered)."""
        action = np.zeros(self.n_dims, dtype=np.int64)
        action[NUM_BUTTONS] = NUM_CAMERA_BINS // 2      # center pitch
        action[NUM_BUTTONS + 1] = NUM_CAMERA_BINS // 2  # center yaw
        return action
