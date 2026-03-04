"""Tests for adaptive_segment_pool1d and GatedAttentionPool modules."""

import torch
import pytest

from src.classification.models import GatedAttentionPool
from src.classification.datamodule import adaptive_segment_pool1d


# ---------------------------------------------------------------------------
# adaptive_segment_pool1d
# ---------------------------------------------------------------------------


class TestAdaptiveSegmentPool1d:
    def test_output_shape(self):
        x = torch.randn(125, 1024)
        out = adaptive_segment_pool1d(x, n_segments=10)
        assert out.shape == (10, 1024)

    def test_fixed_output_regardless_of_input_length(self):
        """Different input lengths produce the same output size."""
        for T in [63, 100, 125, 126]:
            x = torch.randn(T, 64)
            out = adaptive_segment_pool1d(x, n_segments=10)
            assert out.shape == (10, 64), f"Failed for T={T}"

    def test_single_segment(self):
        """n_segments=1 collapses to mean over time."""
        x = torch.randn(50, 16)
        out = adaptive_segment_pool1d(x, n_segments=1)
        assert out.shape == (1, 16)
        expected = x.mean(dim=0, keepdim=True)
        torch.testing.assert_close(out, expected)

    def test_output_size_equals_input(self):
        """When n_segments == T, output equals input."""
        x = torch.randn(10, 32)
        out = adaptive_segment_pool1d(x, n_segments=10)
        assert out.shape == (10, 32)
        torch.testing.assert_close(out, x)

    def test_values_are_segment_means(self):
        """With even division, each output element is the mean of its segment."""
        x = torch.randn(8, 16)
        out = adaptive_segment_pool1d(x, n_segments=2)
        expected_first = x[:4, :].mean(dim=0)
        expected_second = x[4:, :].mean(dim=0)
        torch.testing.assert_close(out[0, :], expected_first)
        torch.testing.assert_close(out[1, :], expected_second)

    def test_gradient_flows(self):
        x = torch.randn(50, 64, requires_grad=True)
        out = adaptive_segment_pool1d(x, n_segments=10)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# GatedAttentionPool
# ---------------------------------------------------------------------------


class TestGatedAttentionPool:
    def test_output_shape(self):
        attn = GatedAttentionPool(1024)
        x = torch.randn(4, 11, 1024)
        out = attn(x)
        assert out.shape == (4, 1024)

    def test_single_timestep(self):
        """With one timestep, output ≈ sigmoid(gate)*x / sigmoid(gate) = x."""
        attn = GatedAttentionPool(64)
        x = torch.randn(2, 1, 64)
        out = attn(x)
        assert out.shape == (2, 64)
        # gate cancels: (gate * x) / gate = x
        torch.testing.assert_close(out, x.squeeze(1), atol=1e-5, rtol=1e-5)

    def test_parameter_count(self):
        d = 256
        attn = GatedAttentionPool(d)
        n_params = sum(p.numel() for p in attn.parameters())
        # Linear(256, 1) = 256 weights + 1 bias = 257
        assert n_params == d + 1

    def test_gradient_flows(self):
        attn = GatedAttentionPool(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_uniform_gates_approximate_mean(self):
        """When gate bias is large, all gates ≈ 1, output ≈ mean pooling."""
        attn = GatedAttentionPool(32)
        # Force gate bias high so sigmoid ≈ 1 for all inputs
        with torch.no_grad():
            attn.gate_fc.weight.zero_()
            attn.gate_fc.bias.fill_(10.0)
        x = torch.randn(3, 20, 32)
        out = attn(x)
        expected = x.mean(dim=1)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_selective_gating(self):
        """Gates with large positive weight on dim 0 attend mostly to high-dim-0 timesteps."""
        attn = GatedAttentionPool(4)
        with torch.no_grad():
            attn.gate_fc.weight.zero_()
            attn.gate_fc.weight[0, 0] = 10.0  # gate keys on dimension 0
            attn.gate_fc.bias.fill_(0.0)
        # Timestep 0: dim 0 = +5 (high gate), timestep 1: dim 0 = -5 (low gate)
        x = torch.zeros(1, 2, 4)
        x[0, 0, 0] = 5.0   # gate ≈ sigmoid(50) ≈ 1.0
        x[0, 0, 1] = 1.0
        x[0, 1, 0] = -5.0  # gate ≈ sigmoid(-50) ≈ 0.0
        x[0, 1, 1] = 99.0  # should be mostly ignored
        out = attn(x)
        # Output dim 1 should be close to 1.0 (from timestep 0), not 99.0
        assert out[0, 1].item() < 2.0
