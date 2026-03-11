"""Mock Minecraft environment for local Mac development.

Provides the same interface as a wrapped MineRL env but with fake data.
Simulates reward signals for testing the full training pipeline.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .action_space import MinecraftActionSpace, NUM_BUTTONS


class MockMinecraftEnv(gym.Env):
    """Fake Minecraft env that returns random observations and shaped rewards.

    Reward logic (for testing):
    - Small positive reward for pressing forward + attack (encourages tree-chopping behavior)
    - Random noise to simulate environment stochasticity
    - Occasional large reward to simulate log drops
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, frame_size: int = 64, max_steps: int = 8000):
        super().__init__()
        self.frame_size = frame_size
        self.max_steps = max_steps
        self.action_space_handler = MinecraftActionSpace()

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_size, frame_size, 3),
            dtype=np.uint8,
        )
        self.action_space = self.action_space_handler.space

        self._step_count = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        obs = self._random_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1

        obs = self._random_obs()
        reward = self._compute_reward(action)
        terminated = self._rng.random() < 0.0005  # rare death
        truncated = self._step_count >= self.max_steps

        info = {"step": self._step_count}
        if terminated:
            info["death"] = True

        return obs, reward, terminated, truncated, info

    def _random_obs(self) -> np.ndarray:
        """Generate a random RGB frame with some structure."""
        obs = self._rng.integers(0, 256, size=(self.frame_size, self.frame_size, 3),
                                 dtype=np.uint8)
        # Add green-ish bottom half to vaguely simulate grass
        obs[self.frame_size // 2:, :, 1] = np.clip(
            obs[self.frame_size // 2:, :, 1].astype(np.int16) + 50, 0, 255
        ).astype(np.uint8)
        return obs

    def _compute_reward(self, action) -> float:
        """Shaped reward to test learning signal."""
        reward = 0.0
        forward = action[0]  # forward button
        attack = action[7]   # attack button

        # Reward forward movement + attack (tree chopping)
        if forward and attack:
            reward += 0.1

        # Random log drop (simulates getting a log item)
        if self._rng.random() < 0.01 and attack:
            reward += 1.0

        # Small noise
        reward += self._rng.normal(0, 0.01)

        return reward

    def render(self):
        return self._random_obs()

    def close(self):
        pass
