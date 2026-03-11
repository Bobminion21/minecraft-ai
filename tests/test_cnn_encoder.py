"""Tests for the IMPALA CNN encoder."""

import torch
import pytest
from minecraft_ai.models.cnn_encoder import IMPALAEncoder


def test_output_shape():
    """(B, 12, 64, 64) -> (B, 256)."""
    encoder = IMPALAEncoder(in_channels=12, feature_dim=256)
    x = torch.randn(4, 12, 64, 64)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_single_sample():
    """Works with batch size 1."""
    encoder = IMPALAEncoder(in_channels=12, feature_dim=256)
    x = torch.randn(1, 12, 64, 64)
    out = encoder(x)
    assert out.shape == (1, 256)


def test_param_count():
    """Should be roughly 130K params (small enough for free Colab)."""
    encoder = IMPALAEncoder(in_channels=12, feature_dim=256)
    n_params = sum(p.numel() for p in encoder.parameters())
    assert 50_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"


def test_gradient_flow():
    """Gradients should flow through the encoder."""
    encoder = IMPALAEncoder(in_channels=12, feature_dim=256)
    x = torch.randn(2, 12, 64, 64)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    for param in encoder.parameters():
        assert param.grad is not None
