"""Plot training curves from CSV logs."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(csv_path: str, output_path: str = None,
                         window: int = 50):
    """Plot training metrics from a CSV log file.

    Args:
        csv_path: path to metrics.csv
        output_path: where to save the plot (shows interactively if None)
        window: smoothing window size
    """
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(val)

    if "step" not in data:
        print("No 'step' column found in CSV.")
        return

    steps = np.array(data["step"])

    # Determine which metrics to plot
    skip_keys = {"step", "elapsed_sec"}
    metric_keys = [k for k in data.keys() if k not in skip_keys]

    if not metric_keys:
        print("No metrics to plot.")
        return

    n_plots = len(metric_keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), squeeze=False)

    for i, key in enumerate(metric_keys):
        ax = axes[i, 0]
        values = np.array(data[key], dtype=float)

        # Raw data (faint)
        ax.plot(steps, values, alpha=0.3, color="blue")

        # Smoothed
        if len(values) >= window:
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], smoothed, color="blue", linewidth=2)

        ax.set_xlabel("Steps")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {output_path}")
    else:
        plt.show()

    plt.close()
