"""All hyperparameters in one dataclass."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Environment
    env_id: str = "MineRLTreechop-v0"
    frame_size: int = 64
    frame_stack: int = 4
    max_episode_steps: int = 8000

    # PPO
    rollout_length: int = 128
    ppo_epochs: int = 4
    num_minibatches: int = 4
    clip_epsilon: float = 0.1
    value_clip: float = 0.1
    max_grad_norm: float = 0.5

    # Optimization
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    adam_eps: float = 1e-5

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Loss coefficients
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Training
    total_timesteps: int = 1_000_000
    checkpoint_interval: int = 50_000
    eval_interval: int = 25_000
    log_interval: int = 1000
    num_eval_episodes: int = 5

    # Curriculum
    curriculum_stages: list = field(default_factory=lambda: [
        {"env_id": "MineRLTreechop-v0", "target_reward": 64.0,
         "name": "TreeChop"},
        {"env_id": "MineRLObtainIronPickaxe-v0", "target_reward": 256.0,
         "name": "IronPickaxe"},
        {"env_id": "MineRLObtainDiamond-v0", "target_reward": 1024.0,
         "name": "Diamond"},
    ])

    # RND (optional curiosity)
    use_rnd: bool = False
    rnd_coeff: float = 0.1
    rnd_feature_dim: int = 128

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    video_dir: str = "videos"

    # Device
    device: Optional[str] = None  # auto-detect if None
    seed: int = 42

    @property
    def minibatch_size(self) -> int:
        return self.rollout_length // self.num_minibatches
