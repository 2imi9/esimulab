"""Fourier Neural Operator (FNO) surrogate for hydrology simulation.

Uses PhysicsNeMo FNO to learn field-to-field mappings:
  Input:  (precipitation, DEM slope, soil type, antecedent moisture)
  Output: (runoff depth, soil moisture)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FNOConfig:
    """Configuration for hydrology FNO surrogate."""

    in_channels: int = 4  # precip, slope, soil_type, moisture
    out_channels: int = 2  # runoff_depth, soil_moisture
    dimension: int = 2
    latent_channels: int = 32
    num_fno_layers: int = 4
    num_fno_modes: int = 16
    padding: int = 8
    activation_fn: str = "gelu"
    resolution: int = 128  # grid resolution for training


@dataclass
class FNOTrainingConfig:
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    checkpoint_dir: str = "checkpoints/fno"
    log_interval: int = 10


def _try_import_physicsnemo():
    """Import PhysicsNeMo FNO, returning None if unavailable."""
    try:
        from physicsnemo.models.fno import FNO

        return FNO
    except ImportError:
        return None


def create_fno_model(config: FNOConfig | None = None) -> Any:
    """Create an FNO model for hydrology surrogate.

    Args:
        config: Model configuration. Defaults to standard hydrology setup.

    Returns:
        PhysicsNeMo FNO model, or a simple PyTorch fallback.
    """
    config = config or FNOConfig()

    fno_cls = _try_import_physicsnemo()
    if fno_cls is not None:
        model = fno_cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            decoder_layers=2,
            decoder_layer_size=32,
            dimension=config.dimension,
            latent_channels=config.latent_channels,
            num_fno_layers=config.num_fno_layers,
            num_fno_modes=config.num_fno_modes,
            padding=config.padding,
            activation_fn=config.activation_fn,
            coord_features=True,
        )
        logger.info(
            "Created PhysicsNeMo FNO: %d→%d channels, %d layers",
            config.in_channels,
            config.out_channels,
            config.num_fno_layers,
        )
        return model

    # Fallback: simple Conv2d surrogate
    logger.warning("PhysicsNeMo not available, creating simple Conv2d fallback")
    return _create_fallback_model(config)


def _create_fallback_model(config: FNOConfig):
    """Create a simple PyTorch Conv2d model as FNO stand-in."""
    import torch.nn as nn

    class SimpleSurrogate(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, out_ch, 1),
            )

        def forward(self, x):
            return self.net(x)

    return SimpleSurrogate(config.in_channels, config.out_channels)


def prepare_training_data(
    dem: np.ndarray,
    precipitation: np.ndarray,
    target_runoff: np.ndarray | None = None,
    target_moisture: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Prepare FNO training data from terrain and atmospheric fields.

    Args:
        dem: Heightfield array (H, W).
        precipitation: Precipitation field (H, W) in mm/hr.
        target_runoff: Ground truth runoff (H, W). None for synthetic.
        target_moisture: Ground truth soil moisture (H, W). None for synthetic.

    Returns:
        Dict with 'input' (C, H, W) and 'target' (C_out, H, W) arrays.
    """
    # Compute slope from DEM
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx**2 + dy**2).astype(np.float32)

    # Normalize inputs
    dem_norm = (dem - dem.mean()) / (dem.std() + 1e-8)
    precip_norm = precipitation / (precipitation.max() + 1e-8)
    slope_norm = slope / (slope.max() + 1e-8)

    # Antecedent moisture (placeholder: uniform)
    moisture = np.full_like(dem, 0.5, dtype=np.float32)

    # Stack input channels: [precip, slope, soil_type_proxy, moisture]
    input_field = np.stack(
        [
            precip_norm,
            slope_norm,
            dem_norm,  # proxy for soil type
            moisture,
        ],
        axis=0,
    ).astype(np.float32)

    # Target: runoff + soil moisture
    if target_runoff is None:
        # Simple synthetic: runoff ∝ precip * slope
        target_runoff = (precip_norm * slope_norm * 10).astype(np.float32)
    if target_moisture is None:
        # Synthetic: moisture increases with precip, decreases with slope
        target_moisture = np.clip(0.5 + precip_norm * 0.3 - slope_norm * 0.2, 0, 1).astype(
            np.float32
        )

    target_field = np.stack([target_runoff, target_moisture], axis=0).astype(np.float32)

    return {"input": input_field, "target": target_field}


def run_inference(
    model: Any,
    input_field: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """Run FNO inference on a single field.

    Args:
        model: FNO or fallback model.
        input_field: (C, H, W) float32 array.
        device: 'cuda' or 'cpu'.

    Returns:
        (C_out, H, W) output array.
    """
    import torch

    with torch.no_grad():
        x = torch.from_numpy(input_field).unsqueeze(0).to(device)  # (1, C, H, W)
        model = model.to(device)
        y = model(x)  # (1, C_out, H, W)
        return y.squeeze(0).cpu().numpy()
