"""Save/load training checkpoints.

Handles:
- Model weights + optimizer state
- Training progress (step count, episode count, curriculum stage)
- Google Drive integration for Colab persistence
"""

import os
import torch
from pathlib import Path
from typing import Optional


class CheckpointManager:
    """Saves and loads training checkpoints."""

    def __init__(self, checkpoint_dir: str, max_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save(self, model, optimizer, step: int, episode: int,
             curriculum_stage: int = 0, extra: dict = None):
        """Save a checkpoint."""
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "episode": episode,
            "curriculum_stage": curriculum_stage,
        }
        if extra:
            state.update(extra)

        path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        torch.save(state, path)

        # Also save a 'latest' symlink-style copy
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(state, latest_path)

        self._cleanup_old()
        return path

    def load(self, model, optimizer=None,
             path: Optional[str] = None, device: torch.device = None):
        """Load a checkpoint. If path is None, loads latest."""
        if path is None:
            path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            path = Path(path)

        if not path.exists():
            return None

        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        return {
            "step": state.get("step", 0),
            "episode": state.get("episode", 0),
            "curriculum_stage": state.get("curriculum_stage", 0),
        }

    def latest_exists(self) -> bool:
        return (self.checkpoint_dir / "checkpoint_latest.pt").exists()

    def _cleanup_old(self):
        """Keep only the most recent max_keep checkpoints (plus latest)."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_[0-9]*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > self.max_keep:
            checkpoints.pop(0).unlink()
