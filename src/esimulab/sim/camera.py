"""Camera configuration and multi-modal rendering for Genesis.

Supports RGB, depth, segmentation, and normal map rendering
via both RayTracer and Rasterizer backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration for Genesis scene."""

    resolution: tuple[int, int] = (1920, 1080)
    fov: float = 45.0
    spp: int = 64  # samples per pixel (RayTracer only)
    denoise: bool = True  # OIDN denoising (Linux, RayTracer)
    model: str = "pinhole"  # 'pinhole' or 'thinlens'
    # Depth-of-field (thinlens only)
    aperture: float = 0.0
    focal_distance: float = 10.0


@dataclass
class RenderOutput:
    """Container for multi-modal render outputs."""

    rgb: np.ndarray | None = None  # (H, W, 3) uint8
    depth: np.ndarray | None = None  # (H, W) float32 meters
    segmentation: np.ndarray | None = None  # (H, W) int32 entity IDs
    normal: np.ndarray | None = None  # (H, W, 3) float32 world-space normals


def setup_camera(
    scene: Any,
    pos: tuple[float, float, float],
    lookat: tuple[float, float, float],
    config: CameraConfig | None = None,
) -> Any:
    """Add a camera to the Genesis scene.

    Args:
        scene: Genesis scene.
        pos: Camera position (x, y, z).
        lookat: Look-at target (x, y, z).
        config: Camera configuration.

    Returns:
        Genesis camera object.
    """
    config = config or CameraConfig()

    camera = scene.add_camera(
        res=config.resolution,
        pos=pos,
        lookat=lookat,
        fov=config.fov,
        spp=config.spp,
    )

    logger.info(
        "Camera added: %dx%d, fov=%d, spp=%d",
        config.resolution[0], config.resolution[1],
        config.fov, config.spp,
    )
    return camera


def render_multimodal(
    camera: Any,
    rgb: bool = True,
    depth: bool = False,
    segmentation: bool = False,
    normal: bool = False,
) -> RenderOutput:
    """Render multiple output modalities from the camera.

    Args:
        camera: Genesis camera object.
        rgb: Render RGB color.
        depth: Render depth map.
        segmentation: Render entity segmentation.
        normal: Render world-space normals.

    Returns:
        RenderOutput with requested modalities.
    """
    try:
        result = camera.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            normal=normal,
        )

        # Genesis returns tuple in order: (rgb, depth, seg, normal)
        output = RenderOutput()
        idx = 0
        if rgb:
            output.rgb = result[idx] if isinstance(result, tuple) else result
            idx += 1
        if depth and isinstance(result, tuple) and idx < len(result):
            output.depth = result[idx]
            idx += 1
        if segmentation and isinstance(result, tuple) and idx < len(result):
            output.segmentation = result[idx]
            idx += 1
        if normal and isinstance(result, tuple) and idx < len(result):
            output.normal = result[idx]

        return output

    except Exception:
        logger.debug("Multi-modal render failed, falling back to RGB only", exc_info=True)
        rgb_data = camera.render()
        return RenderOutput(rgb=rgb_data)


def save_render(
    output: RenderOutput,
    output_dir: str | Path,
    frame_id: str = "frame",
) -> list[Path]:
    """Save render outputs as image files.

    Args:
        output: Multi-modal render output.
        output_dir: Directory to save images.
        frame_id: Filename prefix.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    try:
        from PIL import Image

        if output.rgb is not None:
            path = output_dir / f"{frame_id}_rgb.png"
            Image.fromarray(output.rgb).save(path)
            saved.append(path)

        if output.depth is not None:
            # Normalize depth to 0-255 for visualization
            d = output.depth.astype(np.float32)
            d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
            path = output_dir / f"{frame_id}_depth.png"
            Image.fromarray(d_norm, mode="L").save(path)
            saved.append(path)

            # Also save raw depth as .npy
            np_path = output_dir / f"{frame_id}_depth.npy"
            np.save(np_path, output.depth)
            saved.append(np_path)

        if output.segmentation is not None:
            path = output_dir / f"{frame_id}_seg.npy"
            np.save(path, output.segmentation)
            saved.append(path)

        if output.normal is not None:
            # Map normals from [-1,1] to [0,255]
            n_vis = ((output.normal + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
            path = output_dir / f"{frame_id}_normal.png"
            Image.fromarray(n_vis).save(path)
            saved.append(path)

    except ImportError:
        logger.warning("PIL not available, saving raw numpy arrays only")
        if output.rgb is not None:
            path = output_dir / f"{frame_id}_rgb.npy"
            np.save(path, output.rgb)
            saved.append(path)

    return saved
