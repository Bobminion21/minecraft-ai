"""Environment wrappers: observation normalization, frame stacking, action mapping."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from .action_space import MinecraftActionSpace
from ..utils.config import Config


class FrameStack(gym.Wrapper):
    """Stack the last N frames along the channel dimension.

    Converts (H, W, 3) observations to (H, W, 3*N) by concatenating
    the last N frames. Gives the agent temporal information (motion).
    """

    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        old_space = env.observation_space
        low = np.repeat(old_space.low, num_stack, axis=-1)
        high = np.repeat(old_space.high, num_stack, axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=old_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=-1)


class ResizeObs(gym.ObservationWrapper):
    """Resize observations to a target size using simple slicing/padding.

    For the mock env this is a no-op since it already outputs the right size.
    For MineRL, the POV is typically 64x64 already.
    """

    def __init__(self, env, size: int = 64):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8,
        )

    def observation(self, obs):
        # Simple center crop/pad if needed
        h, w = obs.shape[:2]
        if h == self.size and w == self.size:
            return obs
        # Center crop
        start_h = max(0, (h - self.size) // 2)
        start_w = max(0, (w - self.size) // 2)
        return obs[start_h:start_h + self.size, start_w:start_w + self.size]


def wrap_env(config: Config, env=None):
    """Apply standard wrappers to an environment.

    Args:
        config: training configuration
        env: optional pre-created environment (uses mock if None)

    Returns:
        wrapped gymnasium environment
    """
    if env is None:
        from .mock_env import MockMinecraftEnv
        env = MockMinecraftEnv(
            frame_size=config.frame_size,
            max_steps=config.max_episode_steps,
        )

    env = ResizeObs(env, config.frame_size)
    env = FrameStack(env, config.frame_stack)
    return env
