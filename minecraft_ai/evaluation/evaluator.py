"""Run policy evaluation and collect stats."""

import torch
import numpy as np
from typing import Dict, List


class Evaluator:
    """Evaluate a trained policy over multiple episodes."""

    def __init__(self, env, model, device: torch.device, action_space):
        self.env = env
        self.model = model
        self.device = device
        self.action_space = action_space

    def evaluate(self, num_episodes: int = 5, deterministic: bool = True) -> Dict:
        """Run evaluation episodes and return stats.

        Returns:
            dict with mean_reward, std_reward, mean_length, episode_rewards
        """
        self.model.eval()
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            length = 0

            while not done:
                obs_t = self._obs_to_tensor(obs)
                with torch.no_grad():
                    actions, _, _, _ = self.model.act(obs_t.unsqueeze(0),
                                                      deterministic=deterministic)
                action_np = actions.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                total_reward += reward
                length += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        self.model.train()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "episode_rewards": episode_rewards,
        }

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(obs).float().to(self.device)
        if t.dim() == 3:
            t = t.permute(2, 0, 1)
        t = t / 255.0
        return t
