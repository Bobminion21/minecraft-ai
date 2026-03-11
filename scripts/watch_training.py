"""Live training viewer — watch what the agent sees in real-time.

Controls:
  SPACE  — Pause / Resume training
  S      — Save checkpoint now
  Q/ESC  — Quit and save

The window shows:
  - The agent's current observation (upscaled for visibility)
  - Step count, episode, reward, FPS
  - Pause state
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from minecraft_ai.utils.config import Config
from minecraft_ai.utils.torch_utils import get_device, set_seed
from minecraft_ai.envs.action_space import MinecraftActionSpace
from minecraft_ai.envs.wrappers import wrap_env
from minecraft_ai.models.policy_network import ActorCritic
from minecraft_ai.algo.ppo import PPO
from minecraft_ai.algo.gae import compute_gae
from minecraft_ai.algo.rollout_buffer import RolloutBuffer
from minecraft_ai.training.logger import TrainingLogger
from minecraft_ai.training.checkpoint import CheckpointManager
from minecraft_ai.training.curriculum import CurriculumManager


WINDOW_NAME = "Minecraft AI - Live Training"
DISPLAY_SIZE = 384  # upscale the 64x64 frame for visibility


def obs_to_tensor(obs, device):
    """(H,W,C) uint8 -> (C,H,W) float [0,1]"""
    t = torch.from_numpy(obs).float().to(device)
    if t.dim() == 3:
        t = t.permute(2, 0, 1)
    return t / 255.0


def draw_hud(frame, step, episode, ep_reward, fps, paused, recent_avg):
    """Draw stats overlay on the display frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Text
    color = (0, 255, 0) if not paused else (0, 100, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    line1 = f"Step: {step:,}  Ep: {episode}  FPS: {fps:.0f}"
    line2 = f"Reward: {ep_reward:.2f}  Avg(100): {recent_avg:.2f}"

    cv2.putText(frame, line1, (8, 20), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (8, 42), font, 0.5, color, 1, cv2.LINE_AA)

    if paused:
        # Big PAUSED text in center
        text = "PAUSED (SPACE to resume)"
        size = cv2.getTextSize(text, font, 0.7, 2)[0]
        x = (w - size[0]) // 2
        y = h // 2
        cv2.putText(frame, text, (x, y), font, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

    return frame


def main():
    config = Config(
        total_timesteps=5_000_000,
        rollout_length=128,
        checkpoint_interval=50_000,
        log_interval=1000,
        max_episode_steps=2000,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        video_dir="videos",
        seed=42,
    )

    device = get_device(config.device)
    set_seed(config.seed)
    print(f"Device: {device}")

    # Environment
    env = wrap_env(config)

    # Model + PPO
    action_space = MinecraftActionSpace()
    in_channels = 3 * config.frame_stack
    model = ActorCritic(action_space, in_channels).to(device)
    ppo = PPO(model, config)

    # Buffer
    obs_shape = (in_channels, config.frame_size, config.frame_size)
    buffer = RolloutBuffer(config.rollout_length, obs_shape,
                           action_space.n_dims, device)

    # Logging + checkpointing
    logger = TrainingLogger(config.log_dir)
    ckpt_mgr = CheckpointManager(config.checkpoint_dir)
    curriculum = CurriculumManager(config)

    # Try to resume
    global_step = 0
    episode_count = 0
    if ckpt_mgr.latest_exists():
        info = ckpt_mgr.load(model, ppo.optimizer, device=device)
        if info:
            global_step = info["step"]
            episode_count = info["episode"]
            print(f"Resumed from step {global_step}")

    # State
    episode_reward = 0.0
    recent_rewards = []
    paused = False

    obs, _ = env.reset(seed=config.seed)
    obs_tensor = obs_to_tensor(obs, device)

    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE, DISPLAY_SIZE)

    print("\n" + "=" * 50)
    print("CONTROLS:")
    print("  SPACE  — Pause / Resume")
    print("  S      — Save checkpoint")
    print("  Q/ESC  — Quit and save")
    print("=" * 50 + "\n")

    fps_timer = time.time()
    fps_steps = 0
    fps = 0.0

    try:
        while global_step < config.total_timesteps:
            # --- Display ---
            # Show the most recent single frame (last 3 channels of stacked obs)
            display_frame = obs[:, :, -3:]  # last RGB frame from stack
            display_frame = cv2.resize(display_frame, (DISPLAY_SIZE, DISPLAY_SIZE),
                                       interpolation=cv2.INTER_NEAREST)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            recent_avg = np.mean(recent_rewards[-100:]) if recent_rewards else 0.0
            display_frame = draw_hud(display_frame, global_step, episode_count,
                                     episode_reward, fps, paused, recent_avg)
            cv2.imshow(WINDOW_NAME, display_frame)

            # --- Keyboard input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('s'):
                path = ckpt_mgr.save(model, ppo.optimizer, global_step, episode_count)
                print(f"Manual save: {path}")

            if paused:
                continue

            # --- Collect one step ---
            model.eval()
            with torch.no_grad():
                actions, log_prob, value, _ = model.act(obs_tensor.unsqueeze(0))
                actions = actions.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

            action_np = actions.cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs_tensor, actions, log_prob, reward, value, done)
            global_step += 1
            fps_steps += 1
            episode_reward += reward

            if done:
                recent_rewards.append(episode_reward)
                logger.log_scalar("episode/reward", episode_reward, global_step)
                episode_count += 1
                episode_reward = 0.0
                next_obs, _ = env.reset()

            obs = next_obs
            obs_tensor = obs_to_tensor(obs, device)

            # FPS calculation
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = fps_steps / (now - fps_timer)
                fps_timer = now
                fps_steps = 0

            # --- PPO update when buffer is full ---
            if buffer.is_full():
                model.train()
                with torch.no_grad():
                    _, next_value = model.forward(obs_tensor.unsqueeze(0))
                    next_value = next_value.squeeze(0)

                advantages, returns = compute_gae(
                    buffer.rewards, buffer.values, buffer.dones,
                    next_value, config.gamma, config.gae_lambda,
                )
                buffer.set_advantages(advantages, returns)

                progress = global_step / config.total_timesteps
                ppo.update_learning_rate(progress)
                metrics = ppo.update(buffer)
                buffer.reset()

                if global_step % config.log_interval < config.rollout_length:
                    logger.log_dict(metrics, global_step)
                    logger.log_console(global_step, metrics)

                if global_step % config.checkpoint_interval < config.rollout_length:
                    path = ckpt_mgr.save(model, ppo.optimizer, global_step, episode_count)
                    print(f"Auto-save: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C")

    # Final save
    path = ckpt_mgr.save(model, ppo.optimizer, global_step, episode_count)
    print(f"Final checkpoint: {path}")
    logger.close()
    cv2.destroyAllWindows()
    env.close()

    print(f"\nDone. {global_step:,} steps, {episode_count} episodes.")
    if recent_rewards:
        print(f"Average reward (last 100): {np.mean(recent_rewards[-100:]):.2f}")


if __name__ == "__main__":
    main()
