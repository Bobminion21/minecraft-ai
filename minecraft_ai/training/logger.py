"""Training logger: TensorBoard + CSV + console output."""

import os
import csv
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Logs training metrics to TensorBoard, CSV, and console."""

    def __init__(self, log_dir: str, experiment_name: str = "run"):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self.start_time = time.time()
        self._step = 0

    def log_scalar(self, tag: str, value: float, step: int = None):
        """Log a scalar to TensorBoard."""
        step = step if step is not None else self._step
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict, step: int = None):
        """Log a dict of scalars to TensorBoard and CSV."""
        step = step if step is not None else self._step
        elapsed = time.time() - self.start_time

        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

        # Write to CSV
        row = {"step": step, "elapsed_sec": f"{elapsed:.1f}", **metrics}
        if self.csv_writer is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def log_console(self, step: int, metrics: dict):
        """Print a formatted summary line."""
        elapsed = time.time() - self.start_time
        fps = step / elapsed if elapsed > 0 else 0
        parts = [f"Step {step:>8d} | {elapsed:>6.0f}s | {fps:>5.0f} fps"]
        for key, val in metrics.items():
            if isinstance(val, float):
                parts.append(f"{key}: {val:.4f}")
            else:
                parts.append(f"{key}: {val}")
        print(" | ".join(parts))

    def set_step(self, step: int):
        self._step = step

    def close(self):
        self.writer.close()
        if self.csv_file:
            self.csv_file.close()
