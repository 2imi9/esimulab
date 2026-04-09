"""Tests for PINO physics-informed training utilities."""

import numpy as np
import torch

from esimulab.surrogate.pino import (
    PINOConfig,
    compute_shallow_water_residual,
    pino_loss,
)


class TestShallowWaterResidual:
    def test_uniform_fields_zero_residual(self):
        h = np.ones((32, 32), dtype=np.float32)
        u = np.zeros((32, 32), dtype=np.float32)
        v = np.zeros((32, 32), dtype=np.float32)
        residuals = compute_shallow_water_residual(h, u, v, dx=1.0, dy=1.0)

        assert "continuity" in residuals
        assert "x_momentum" in residuals
        assert "y_momentum" in residuals
        np.testing.assert_allclose(residuals["continuity"], 0.0, atol=1e-5)

    def test_returns_correct_shape(self):
        h = np.random.rand(16, 16).astype(np.float32) + 0.1
        u = np.random.rand(16, 16).astype(np.float32)
        v = np.random.rand(16, 16).astype(np.float32)
        residuals = compute_shallow_water_residual(h, u, v, dx=10.0, dy=10.0)

        assert residuals["continuity"].shape == (16, 16)
        assert residuals["x_momentum"].shape == (16, 16)


class TestPINOLoss:
    def test_data_only_loss(self):
        pred = torch.randn(2, 2, 16, 16)
        target = torch.randn(2, 2, 16, 16)
        config = PINOConfig(pde_weight=0.0)

        losses = pino_loss(pred, target, config)
        assert losses["total"] > 0
        assert losses["pde"].item() == 0.0

    def test_with_pde_loss(self):
        pred = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        config = PINOConfig(pde_weight=0.1)

        losses = pino_loss(pred, target, config)
        assert losses["total"] > 0
        assert losses["pde"] > 0

    def test_default_config(self):
        config = PINOConfig()
        assert config.pde_weight == 0.1
        assert config.gradient_method == "fdm"
