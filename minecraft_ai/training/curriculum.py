"""Curriculum manager: advance through progressively harder tasks."""

from ..utils.config import Config


class CurriculumManager:
    """Tracks curriculum stage and decides when to advance."""

    def __init__(self, config: Config):
        self.stages = config.curriculum_stages
        self.current_stage = 0
        self._window_size = 100  # episodes to average over

    @property
    def current_stage_name(self) -> str:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]["name"]
        return "COMPLETED"

    @property
    def current_env_id(self) -> str:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]["env_id"]
        return self.stages[-1]["env_id"]

    @property
    def target_reward(self) -> float:
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]["target_reward"]
        return float("inf")

    def maybe_advance(self, avg_reward: float) -> bool:
        """Check if we should advance to the next curriculum stage.

        Args:
            avg_reward: rolling average episode reward

        Returns:
            True if advanced to a new stage
        """
        if self.current_stage >= len(self.stages) - 1:
            return False

        if avg_reward >= self.target_reward:
            self.current_stage += 1
            return True

        return False
