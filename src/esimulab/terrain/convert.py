"""Convert DEM data to Genesis-ready heightfield format."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenesisHeightfield:
    """Genesis-ready heightfield data."""

    height_field: np.ndarray  # float32, NaN-free
    horizontal_scale: float  # meters per pixel
    vertical_scale: float  # z-axis multiplier
    origin: tuple[float, float, float]  # (x, y, z) world position
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


def prepare_heightfield(
    dem: np.ndarray,
    pixel_size: float,
    vertical_scale: float = 1.0,
    fill_nodata: float = 0.0,
    center_origin: bool = True,
) -> GenesisHeightfield:
    """Prepare a DEM array for Genesis terrain loading.

    Args:
        dem: float32 heightfield array (rows, cols). May contain NaN for nodata.
        pixel_size: Meters per pixel in the projected CRS.
        vertical_scale: Multiplier for Z values. Use >1 for exaggeration.
        fill_nodata: Value to replace NaN pixels with.
        center_origin: If True, center the terrain at (0, 0) in XY.

    Returns:
        GenesisHeightfield with cleaned array and placement metadata.
    """
    hf = np.asarray(dem, dtype=np.float32).copy()

    nan_count = int(np.isnan(hf).sum())
    if nan_count > 0:
        logger.info("Filling %d nodata pixels with %.1f", nan_count, fill_nodata)
        hf = np.nan_to_num(hf, nan=fill_nodata)

    z_min = float(hf.min())
    z_max = float(hf.max())
    rows, cols = hf.shape
    x_extent = cols * pixel_size
    y_extent = rows * pixel_size

    origin = (-x_extent / 2, -y_extent / 2, 0.0) if center_origin else (0.0, 0.0, 0.0)

    bounds_min = (origin[0], origin[1], z_min * vertical_scale)
    bounds_max = (origin[0] + x_extent, origin[1] + y_extent, z_max * vertical_scale)

    logger.info(
        "Heightfield: %dx%d, z=[%.1f, %.1f]m, extent=%.0fx%.0fm",
        cols, rows, z_min, z_max, x_extent, y_extent,
    )

    return GenesisHeightfield(
        height_field=hf,
        horizontal_scale=pixel_size,
        vertical_scale=vertical_scale,
        origin=origin,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )
