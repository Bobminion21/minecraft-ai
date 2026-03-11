"""IMPALA-style residual CNN encoder.

Architecture: 3 stages of [Conv -> MaxPool -> ResBlock -> ResBlock]
Input:  (B, C, 64, 64) where C = 3 * frame_stack (e.g., 12 for 4 frames)
Output: (B, 256) feature vector

~130K parameters -- small enough for free Colab T4 GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_utils import init_weights


class ResidualBlock(nn.Module):
    """Simple residual block: two 3x3 convs with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class ConvSequence(nn.Module):
    """One IMPALA stage: Conv -> MaxPool -> ResBlock -> ResBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class IMPALAEncoder(nn.Module):
    """IMPALA-style CNN: 3 ConvSequence stages -> flatten -> Linear -> 256.

    Channel progression: in_channels -> 16 -> 32 -> 32
    Spatial reduction: 64 -> 32 -> 16 -> 8 (three 2x max pools)
    Flattened: 32 * 8 * 8 = 2048 -> Linear -> 256
    """

    def __init__(self, in_channels: int = 12, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.stages = nn.Sequential(
            ConvSequence(in_channels, 16),
            ConvSequence(16, 32),
            ConvSequence(32, 32),
        )
        # After 3 stages of stride-2 pooling: 64 -> 32 -> 16 -> 8
        self.fc = nn.Linear(32 * 8 * 8, feature_dim)

        self.apply(lambda m: init_weights(m, gain=1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 64, 64) float tensor, normalized to [0, 1]
        Returns:
            (B, feature_dim) feature vector
        """
        x = self.stages(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)  # flatten
        x = self.fc(x)
        x = F.relu(x)
        return x
