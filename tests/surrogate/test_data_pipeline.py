"""Tests for FNO training data pipeline."""

import numpy as np

from esimulab.surrogate.data_pipeline import (
    create_dataloader,
    generate_synthetic_training_data,
    particles_to_grid,
    train_fno_surrogate,
)


class TestSyntheticData:
    def test_generates_correct_shapes(self):
        data = generate_synthetic_training_data(num_samples=10, resolution=32)
        assert data["inputs"].shape == (10, 4, 32, 32)
        assert data["targets"].shape == (10, 2, 32, 32)

    def test_values_normalized(self):
        data = generate_synthetic_training_data(num_samples=5, resolution=16)
        assert data["inputs"].max() <= 1.0 + 1e-5
        assert data["targets"][:, 1].max() <= 1.0 + 1e-5  # soil moisture

    def test_deterministic(self):
        d1 = generate_synthetic_training_data(num_samples=3, resolution=16, seed=42)
        d2 = generate_synthetic_training_data(num_samples=3, resolution=16, seed=42)
        np.testing.assert_array_equal(d1["inputs"], d2["inputs"])


class TestParticlesToGrid:
    def test_basic_grid(self):
        particles = np.array([[50, 50, 0], [50, 50, 0]], dtype=np.float32)
        grid = particles_to_grid(particles, (0, 0, 100, 100), resolution=(10, 10))
        assert grid.shape == (10, 10)
        assert grid.sum() == 2.0

    def test_empty_particles(self):
        particles = np.zeros((0, 3), dtype=np.float32)
        grid = particles_to_grid(particles, (0, 0, 100, 100))
        assert grid.sum() == 0.0


class TestDataLoader:
    def test_creates_loader(self):
        data = generate_synthetic_training_data(num_samples=20, resolution=16)
        loader = create_dataloader(data, batch_size=4, train=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] <= 4
        assert batch[0].shape[1] == 4  # input channels

    def test_train_val_split(self):
        data = generate_synthetic_training_data(num_samples=100, resolution=8)
        train = create_dataloader(data, batch_size=8, train=True, train_split=0.8)
        val = create_dataloader(data, batch_size=8, train=False, train_split=0.8)
        assert len(train.dataset) == 80
        assert len(val.dataset) == 20


class TestTraining:
    def test_short_training_run(self):
        from esimulab.surrogate.fno import create_fno_model

        model = create_fno_model()
        data = generate_synthetic_training_data(num_samples=20, resolution=16)
        history = train_fno_surrogate(
            model, data, epochs=2, device="cpu"
        )
        assert len(history["train_losses"]) == 2
        assert len(history["val_losses"]) == 2
        assert all(loss > 0 for loss in history["train_losses"])
