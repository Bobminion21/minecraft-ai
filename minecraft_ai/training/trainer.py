"""Main training loop: collect rollouts -> GAE -> PPO update -> repeat."""

import torch
import numpy as np
from typing import Optional

from ..utils.config import Config
from ..utils.torch_utils import get_device, set_seed
from ..envs.action_space import MinecraftActionSpace
from ..envs.wrappers import wrap_env
from ..models.policy_network import ActorCritic
from ..algo.ppo import PPO
from ..algo.gae import compute_gae
from ..algo.rollout_buffer import RolloutBuffer
from .logger import TrainingLogger
from .checkpoint import CheckpointManager
from .curriculum import CurriculumManager


class Trainer:
    """Orchestrates the full PPO training pipeline."""

    def __init__(self, config: Config, env=None):
        self.config = config
        self.device = get_device(config.device)
        set_seed(config.seed)

        # Environment
        if env is not None:
            self.env = env
        else:
            self.env = wrap_env(config)

        # Action space
        self.action_space = MinecraftActionSpace()

        # Model
        in_channels = 3 * config.frame_stack
        self.model = ActorCritic(self.action_space, in_channels).to(self.device)

        # PPO
        self.ppo = PPO(self.model, config)

        # Rollout buffer
        obs_shape = (in_channels, config.frame_size, config.frame_size)
        self.buffer = RolloutBuffer(
            config.rollout_length, obs_shape,
            self.action_space.n_dims, self.device,
        )

        # Logging + Checkpointing
        self.logger = TrainingLogger(config.log_dir)
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        self.curriculum = CurriculumManager(config)

        # State
        self.global_step = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.recent_rewards = []

    def train(self):
        """Main training loop."""
        print(f"Training on {self.device} | Target: {self.config.total_timesteps} steps")
        print(f"Curriculum stage: {self.curriculum.current_stage_name}")

        # Try to resume from checkpoint
        self._maybe_resume()

        # Initial reset
        obs, _ = self.env.reset(seed=self.config.seed)
        obs = self._obs_to_tensor(obs)

        while self.global_step < self.config.total_timesteps:
            # Collect rollout
            obs = self._collect_rollout(obs)

            # Compute advantages
            with torch.no_grad():
                # Get value of last observation for bootstrapping
                _, next_value = self.model.forward(obs.unsqueeze(0))
                next_value = next_value.squeeze(0)

            advantages, returns = compute_gae(
                self.buffer.rewards, self.buffer.values, self.buffer.dones,
                next_value, self.config.gamma, self.config.gae_lambda,
            )
            self.buffer.set_advantages(advantages, returns)

            # PPO update
            progress = self.global_step / self.config.total_timesteps
            self.ppo.update_learning_rate(progress)
            metrics = self.ppo.update(self.buffer)
            self.buffer.reset()

            # Logging
            if self.global_step % self.config.log_interval < self.config.rollout_length:
                self.logger.log_dict(metrics, self.global_step)
                self.logger.log_console(self.global_step, metrics)

            # Checkpointing
            if self.global_step % self.config.checkpoint_interval < self.config.rollout_length:
                self._save_checkpoint()

            # Curriculum check
            if self.recent_rewards:
                avg_reward = np.mean(self.recent_rewards[-100:])
                advanced = self.curriculum.maybe_advance(avg_reward)
                if advanced:
                    print(f"Advanced to: {self.curriculum.current_stage_name}")
                    self.logger.log_scalar(
                        "curriculum/stage", self.curriculum.current_stage, self.global_step
                    )

        # Final checkpoint
        self._save_checkpoint()
        self.logger.close()
        print(f"Training complete. {self.global_step} steps, {self.episode_count} episodes.")

    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """Collect rollout_length steps of experience."""
        self.model.eval()

        for _ in range(self.config.rollout_length):
            with torch.no_grad():
                actions, log_prob, value, _ = self.model.act(obs.unsqueeze(0))
                actions = actions.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

            # Step environment
            action_np = actions.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store transition
            self.buffer.add(obs, actions, log_prob, reward, value, done)
            self.global_step += 1

            # Track episode stats
            self.episode_reward += reward
            self.episode_length += 1

            if done:
                self.recent_rewards.append(self.episode_reward)
                self.logger.log_scalar("episode/reward", self.episode_reward, self.global_step)
                self.logger.log_scalar("episode/length", self.episode_length, self.global_step)
                self.episode_count += 1
                self.episode_reward = 0.0
                self.episode_length = 0
                next_obs, _ = self.env.reset()

            obs = self._obs_to_tensor(next_obs)

        self.model.train()
        return obs

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor: (H,W,C) uint8 -> (C,H,W) float [0,1]."""
        t = torch.from_numpy(obs).float().to(self.device)
        if t.dim() == 3:
            t = t.permute(2, 0, 1)  # HWC -> CHW
        t = t / 255.0
        return t

    def _save_checkpoint(self):
        path = self.checkpoint_mgr.save(
            self.model, self.ppo.optimizer,
            self.global_step, self.episode_count,
            self.curriculum.current_stage,
        )
        print(f"Checkpoint saved: {path}")

    def _maybe_resume(self):
        if self.checkpoint_mgr.latest_exists():
            info = self.checkpoint_mgr.load(
                self.model, self.ppo.optimizer, device=self.device,
            )
            if info:
                self.global_step = info["step"]
                self.episode_count = info["episode"]
                self.curriculum.current_stage = info["curriculum_stage"]
                print(f"Resumed from step {self.global_step}")
