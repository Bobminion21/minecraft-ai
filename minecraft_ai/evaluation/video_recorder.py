"""Record agent gameplay as MP4 video."""

import torch
import numpy as np
from pathlib import Path

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio


class VideoRecorder:
    """Record an agent playing and save as MP4."""

    def __init__(self, env, model, device: torch.device, video_dir: str = "videos"):
        self.env = env
        self.model = model
        self.device = device
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def record(self, filename: str = "gameplay.mp4", max_steps: int = 2000,
               fps: int = 20, deterministic: bool = True) -> str:
        """Record one episode of gameplay.

        Args:
            filename: output filename
            max_steps: max frames to record
            fps: video framerate
            deterministic: use greedy policy

        Returns:
            path to saved video
        """
        self.model.eval()
        frames = []

        obs, _ = self.env.reset()
        for step in range(max_steps):
            # Get a render frame (use the raw observation before frame stacking)
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)

            obs_t = self._obs_to_tensor(obs)
            with torch.no_grad():
                actions, _, _, _ = self.model.act(obs_t.unsqueeze(0),
                                                   deterministic=deterministic)
            action_np = actions.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(action_np)

            if terminated or truncated:
                break

        self.model.train()

        # Save video
        path = self.video_dir / filename
        if frames:
            iio.imwrite(str(path), np.stack(frames), fps=fps,
                        codec="libx264", plugin="pyav")
            print(f"Video saved: {path} ({len(frames)} frames)")
        else:
            print("No frames captured.")

        return str(path)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(obs).float().to(self.device)
        if t.dim() == 3:
            t = t.permute(2, 0, 1)
        t = t / 255.0
        return t
