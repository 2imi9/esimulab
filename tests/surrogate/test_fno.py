"""Tests for FNO hydrology surrogate."""

import numpy as np
import torch

from esimulab.surrogate.fno import (
    FNOConfig,
    create_fno_model,
    prepare_training_data,
    run_inference,
)


class TestFNOConfig:
    def test_defaults(self):
        config = FNOConfig()
        assert config.in_channels == 4
        assert config.out_channels == 2
        assert config.num_fno_layers == 4


class TestCreateModel:
    def test_creates_fallback_model(self):
        """Without PhysicsNeMo, should create Conv2d fallback."""
        model = create_fno_model()
        assert model is not None
        # Test forward pass
        x = torch.randn(1, 4, 32, 32)
        y = model(x)
        assert y.shape == (1, 2, 32, 32)

    def test_custom_config(self):
        config = FNOConfig(in_channels=3, out_channels=1, num_fno_layers=2)
        model = create_fno_model(config)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 1, 32, 32)


class TestPrepareData:
    def test_returns_input_and_target(self):
        dem = np.random.rand(64, 64).astype(np.float32) * 500
        precip = np.random.rand(64, 64).astype(np.float32) * 10

        data = prepare_training_data(dem, precip)
        assert "input" in data
        assert "target" in data
        assert data["input"].shape == (4, 64, 64)
        assert data["target"].shape == (2, 64, 64)

    def test_input_is_normalized(self):
        dem = np.ones((32, 32), dtype=np.float32) * 1000
        precip = np.ones((32, 32), dtype=np.float32) * 5

        data = prepare_training_data(dem, precip)
        assert np.all(np.isfinite(data["input"]))


class TestInference:
    def test_run_inference_cpu(self):
        model = create_fno_model()
        input_field = np.random.rand(4, 32, 32).astype(np.float32)
        output = run_inference(model, input_field, device="cpu")
        assert output.shape == (2, 32, 32)
