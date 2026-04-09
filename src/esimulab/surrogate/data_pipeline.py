"""Training data pipeline for FNO/MeshGraphNet surrogates.

Generates training pairs from:
1. WRF-Hydro simulation grids (streamflow, soil moisture, water table)
2. Genesis simulation output (particle positions → gridded fields)
3. Synthetic data generation for testing/prototyping

The Darcy2D datapipe from PhysicsNeMo serves as the template for
custom field-to-field surrogate training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataConfig:
    """Configuration for training data generation."""

    grid_resolution: int = 128
    num_samples: int = 1000
    train_split: float = 0.8
    normalize: bool = True
    augment: bool = True  # random flips, rotations
    seed: int = 42


def generate_synthetic_training_data(
    num_samples: int = 100,
    resolution: int = 128,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic terrain→hydrology training pairs.

    Creates paired (input, target) fields simulating:
    - Input: precipitation + terrain slope + soil type + moisture
    - Target: runoff depth + soil moisture response

    Physics-inspired synthetic rules:
    - Runoff increases with precipitation and slope
    - Soil moisture increases with precipitation, decreases with slope
    - Channel formation follows steepest descent paths

    Args:
        num_samples: Number of training samples.
        resolution: Grid resolution (H x W).
        seed: Random seed.

    Returns:
        Dict with 'inputs' (N, 4, H, W) and 'targets' (N, 2, H, W).
    """
    rng = np.random.default_rng(seed)
    h, w = resolution, resolution

    inputs = np.zeros((num_samples, 4, h, w), dtype=np.float32)
    targets = np.zeros((num_samples, 2, h, w), dtype=np.float32)

    for i in range(num_samples):
        # Generate terrain (multi-scale Perlin-like noise)
        terrain = _generate_terrain(h, w, rng)
        dy, dx = np.gradient(terrain)
        slope = np.sqrt(dx**2 + dy**2)

        # Precipitation field (spatially varying)
        precip = rng.exponential(2.0, (h, w)).astype(np.float32)
        precip *= rng.uniform(0.5, 2.0)  # scale variation

        # Soil type proxy (correlated with terrain)
        soil_type = (terrain > np.median(terrain)).astype(np.float32) * 0.5 + 0.25

        # Antecedent moisture
        moisture = rng.uniform(0.2, 0.8, (h, w)).astype(np.float32)

        # Normalize inputs
        inputs[i, 0] = precip / (precip.max() + 1e-8)
        inputs[i, 1] = slope / (slope.max() + 1e-8)
        inputs[i, 2] = soil_type
        inputs[i, 3] = moisture

        # Compute targets using simplified physics
        # Runoff: SCS curve number-inspired
        cn = 60 + soil_type * 30  # curve number
        s = 25400 / cn - 254  # retention
        ia = 0.2 * s  # initial abstraction
        pe = precip - ia
        pe = np.maximum(pe, 0)
        runoff = pe**2 / (pe + s + 1e-8)
        runoff *= 1 + slope * 2  # slope enhancement

        # Soil moisture response
        soil_moisture = np.clip(
            moisture + precip * 0.05 - slope * 0.1 - runoff * 0.02,
            0,
            1,
        )

        targets[i, 0] = runoff / (runoff.max() + 1e-8)
        targets[i, 1] = soil_moisture

    logger.info(
        "Generated %d training samples: inputs=%s, targets=%s",
        num_samples,
        inputs.shape,
        targets.shape,
    )
    return {"inputs": inputs, "targets": targets}


def _generate_terrain(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic terrain using multi-scale noise."""
    terrain = np.zeros((h, w), dtype=np.float32)

    for scale in [4, 8, 16, 32]:
        noise = rng.normal(0, 1, (h // scale + 1, w // scale + 1)).astype(np.float32)
        # Bilinear upsample
        from scipy.ndimage import zoom

        upsampled = zoom(noise, (h / (h // scale + 1), w / (w // scale + 1)), order=1)
        terrain += upsampled[:h, :w] * (scale / 32)

    # Normalize to 0-1000m range
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    terrain *= 1000
    return terrain


def particles_to_grid(
    particles: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution: tuple[int, int] = (128, 128),
    value_channel: int | None = None,
) -> np.ndarray:
    """Convert particle positions to gridded density/value field.

    Used to convert Genesis SPH output to training targets for FNO.

    Args:
        particles: (N, 3+) array with at least xyz positions.
        bounds: (xmin, ymin, xmax, ymax) spatial bounds.
        resolution: Output grid resolution.
        value_channel: If specified, use this column as value weight.

    Returns:
        (H, W) float32 grid.
    """
    xmin, ymin, xmax, ymax = bounds
    h, w = resolution

    x = particles[:, 0]
    y = particles[:, 1]

    if value_channel is not None and particles.shape[1] > value_channel:
        weights = particles[:, value_channel]
    else:
        weights = np.ones(len(x), dtype=np.float32)

    grid, _, _ = np.histogram2d(
        y,
        x,
        bins=[h, w],
        range=[[ymin, ymax], [xmin, xmax]],
        weights=weights,
    )

    return grid.astype(np.float32)


def create_dataloader(
    data: dict[str, np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    train: bool = True,
    train_split: float = 0.8,
) -> Any:
    """Create a PyTorch DataLoader from training data.

    Args:
        data: Dict with 'inputs' and 'targets' arrays.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        train: If True, use training split; else validation split.
        train_split: Fraction for training.

    Returns:
        PyTorch DataLoader.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    inputs = torch.from_numpy(data["inputs"])
    targets = torch.from_numpy(data["targets"])

    n = len(inputs)
    split = int(n * train_split)

    if train:
        dataset = TensorDataset(inputs[:split], targets[:split])
    else:
        dataset = TensorDataset(inputs[split:], targets[split:])

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fno_surrogate(
    model: Any,
    train_data: dict[str, np.ndarray],
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    checkpoint_dir: str | Path | None = None,
) -> dict[str, list[float]]:
    """Train an FNO model on hydrology data.

    Args:
        model: FNO or fallback model.
        train_data: Dict with 'inputs' (N,4,H,W) and 'targets' (N,2,H,W).
        epochs: Training epochs.
        learning_rate: Optimizer learning rate.
        device: 'cuda' or 'cpu'.
        checkpoint_dir: Directory for saving checkpoints.

    Returns:
        Dict with 'train_losses' and 'val_losses' per epoch.
    """
    import torch

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    train_loader = create_dataloader(train_data, batch_size=32, train=True)
    val_loader = create_dataloader(train_data, batch_size=32, train=False, shuffle=False)

    history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_losses"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                val_loss += criterion(output, targets).item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        history["val_losses"].append(val_loss)

        scheduler.step()

        if epoch % 10 == 0:
            logger.info(
                "Epoch %d/%d: train=%.6f, val=%.6f, lr=%.6f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                scheduler.get_last_lr()[0],
            )

    # Save checkpoint
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "fno_best.pt")
        logger.info("Checkpoint saved to %s", ckpt_path / "fno_best.pt")

    return history
