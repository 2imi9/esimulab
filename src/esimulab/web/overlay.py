"""Generate image overlays from simulation data for CesiumJS imagery layers.

Converts simulation output (wind fields, particle density, runoff depth)
into georeferenced PNG images that CesiumJS can display as imagery overlays.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_rgba_png(data: np.ndarray, colormap: str = "viridis") -> bytes:
    """Convert a 2D array to RGBA PNG bytes using matplotlib colormap.

    Args:
        data: 2D float array (will be normalized to 0-1).
        colormap: Matplotlib colormap name.

    Returns:
        PNG image bytes.
    """
    try:
        from matplotlib import cm
        from PIL import Image

        # Normalize to 0-1
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        rng = vmax - vmin
        normalized = (data - vmin) / rng if rng > 0 else np.zeros_like(data)

        normalized = np.nan_to_num(normalized, nan=0.0)

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        rgba = (cmap(normalized) * 255).astype(np.uint8)

        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        logger.warning("matplotlib/PIL not available for overlay generation")
        return b""


def generate_wind_overlay(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    output_path: str | Path | None = None,
) -> bytes:
    """Generate wind speed overlay image.

    Args:
        wind_u: 2D eastward wind component.
        wind_v: 2D northward wind component.
        output_path: Optional file path to save PNG.

    Returns:
        PNG bytes of wind speed colormap.
    """
    speed = np.sqrt(wind_u**2 + wind_v**2)
    png_bytes = _to_rgba_png(speed, colormap="YlOrRd")

    if output_path and png_bytes:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(png_bytes)
        logger.info("Wind overlay saved: %s", output_path)

    return png_bytes


def generate_precip_overlay(
    precipitation: np.ndarray,
    output_path: str | Path | None = None,
) -> bytes:
    """Generate precipitation overlay image.

    Args:
        precipitation: 2D precipitation field (mm/hr).
        output_path: Optional file path to save PNG.

    Returns:
        PNG bytes of precipitation colormap.
    """
    png_bytes = _to_rgba_png(precipitation, colormap="Blues")

    if output_path and png_bytes:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(png_bytes)

    return png_bytes


def generate_particle_density_overlay(
    particles: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution: tuple[int, int] = (256, 256),
    output_path: str | Path | None = None,
) -> bytes:
    """Generate particle density heatmap overlay.

    Args:
        particles: (N, 3) particle positions.
        bounds: (xmin, ymin, xmax, ymax) spatial bounds.
        resolution: Output image resolution (width, height).
        output_path: Optional file path to save PNG.

    Returns:
        PNG bytes of density heatmap.
    """
    xmin, ymin, xmax, ymax = bounds
    w, h = resolution

    # 2D histogram of particle positions
    x = particles[:, 0]
    y = particles[:, 1]
    density, _, _ = np.histogram2d(
        y, x,
        bins=[h, w],
        range=[[ymin, ymax], [xmin, xmax]],
    )

    # Log scale for better visualization
    density = np.log1p(density).astype(np.float32)

    png_bytes = _to_rgba_png(density, colormap="hot")

    if output_path and png_bytes:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(png_bytes)

    return png_bytes


def overlay_metadata(
    bbox: tuple[float, float, float, float],
    overlay_type: str,
) -> dict:
    """Generate CesiumJS-compatible metadata for an overlay.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        overlay_type: Type identifier ('wind', 'precip', 'density').

    Returns:
        Dict with CesiumJS imagery layer configuration.
    """
    west, south, east, north = bbox
    return {
        "type": overlay_type,
        "bounds": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "url": f"/api/overlays/{overlay_type}.png",
        "alpha": 0.6,
        "cesium_config": {
            "rectangle": {
                "west": west * np.pi / 180,
                "south": south * np.pi / 180,
                "east": east * np.pi / 180,
                "north": north * np.pi / 180,
            },
        },
    }
