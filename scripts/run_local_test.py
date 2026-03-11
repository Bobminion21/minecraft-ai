"""Smoke test: run a short training session with the mock environment.

This verifies the full pipeline works end-to-end on Mac without MineRL.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minecraft_ai.utils.config import Config
from minecraft_ai.training.trainer import Trainer


def main():
    config = Config(
        total_timesteps=512,       # Just 4 rollouts (128 * 4)
        rollout_length=128,
        checkpoint_interval=256,
        log_interval=128,
        eval_interval=512,
        max_episode_steps=200,
        checkpoint_dir="/tmp/mc_ai_test/checkpoints",
        log_dir="/tmp/mc_ai_test/logs",
        video_dir="/tmp/mc_ai_test/videos",
        seed=42,
    )

    print("=" * 60)
    print("Minecraft AI - Local Smoke Test")
    print(f"  Steps: {config.total_timesteps}")
    print(f"  Rollout: {config.rollout_length}")
    print(f"  Device: auto-detect")
    print("=" * 60)

    trainer = Trainer(config)
    trainer.train()

    print("\nSmoke test PASSED!")
    print(f"  Episodes completed: {trainer.episode_count}")
    print(f"  Final step: {trainer.global_step}")

    # Quick eval
    if trainer.recent_rewards:
        import numpy as np
        print(f"  Mean episode reward: {np.mean(trainer.recent_rewards):.2f}")


if __name__ == "__main__":
    main()
