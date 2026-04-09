"""Renderer selection and RayTracer configuration for Genesis.

RayTracer requires LuisaRenderPy (Linux-only C++ renderer).
This module handles automatic fallback, configuration, and
Docker-aware renderer selection.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RayTracerConfig:
    """Configuration for Genesis RayTracer (photorealistic rendering)."""

    tracing_depth: int = 32
    rr_depth: int = 0  # Russian Roulette start depth
    rr_threshold: float = 0.95
    spp: int = 64  # samples per pixel (preview)
    spp_final: int = 256  # samples per pixel (final render)
    denoise: bool = True  # OIDN denoising (Linux only)
    lights: list[dict] | None = None

    def default_lights(self, z_top: float = 200.0) -> list[dict]:
        """Generate default lighting setup for terrain."""
        return self.lights or [
            {"pos": (0, 0, z_top + 100), "color": (1.0, 0.98, 0.9), "intensity": 12, "radius": 5},
            {
                "pos": (500, -300, z_top + 200),
                "color": (0.9, 0.95, 1.0),
                "intensity": 4,
                "radius": 8,
            },
        ]


@dataclass
class RasterizerConfig:
    """Configuration for Genesis Rasterizer (real-time rendering)."""

    # Rasterizer is always available, no special config needed
    pass


def create_renderer(
    gs: Any,
    raytracer_config: RayTracerConfig | None = None,
    force_rasterizer: bool = False,
    z_top: float = 200.0,
) -> tuple[Any, str]:
    """Create the best available renderer for the current platform.

    Selection logic:
    1. If force_rasterizer=True → Rasterizer
    2. If Windows/macOS → Rasterizer (LuisaRender is Linux-only)
    3. Try RayTracer → if LuisaRenderPy missing → Rasterizer

    Args:
        gs: Genesis module.
        raytracer_config: RayTracer configuration (ignored if Rasterizer selected).
        force_rasterizer: Skip RayTracer attempt.
        z_top: Terrain top elevation for light placement.

    Returns:
        Tuple of (renderer_instance, renderer_name).
    """
    if force_rasterizer:
        logger.info("Rasterizer forced by configuration")
        return gs.renderers.Rasterizer(), "rasterizer"

    if platform.system() != "Linux":
        logger.info(
            "Non-Linux platform (%s) — using Rasterizer. "
            "RayTracer requires LuisaRenderPy which is Linux-only. "
            "Use Docker for photorealistic rendering.",
            platform.system(),
        )
        return gs.renderers.Rasterizer(), "rasterizer"

    # Linux: try RayTracer
    config = raytracer_config or RayTracerConfig()
    try:
        renderer = gs.renderers.RayTracer(
            tracing_depth=config.tracing_depth,
            rr_depth=config.rr_depth,
            rr_threshold=config.rr_threshold,
            lights=config.default_lights(z_top),
        )
        logger.info(
            "RayTracer initialized: depth=%d, spp=%d, denoise=%s",
            config.tracing_depth,
            config.spp,
            config.denoise,
        )
        return renderer, "raytracer"

    except Exception as e:
        logger.warning(
            "RayTracer failed (%s). Install LuisaRenderPy: "
            "pip install LuisaRenderPy (Linux only, requires CUDA). "
            "Falling back to Rasterizer.",
            e,
        )
        return gs.renderers.Rasterizer(), "rasterizer"


def check_luisa_render() -> dict[str, Any]:
    """Check LuisaRender availability and system compatibility.

    Returns:
        Dict with 'available', 'platform', 'reason'.
    """
    info: dict[str, Any] = {
        "platform": platform.system(),
        "available": False,
        "reason": "",
    }

    if platform.system() != "Linux":
        info["reason"] = f"LuisaRenderPy requires Linux (current: {platform.system()})"
        return info

    try:
        import LuisaRenderPy  # noqa: F401

        info["available"] = True
        info["reason"] = "LuisaRenderPy installed and available"
    except ImportError:
        info["reason"] = (
            "LuisaRenderPy not installed. Install with: pip install LuisaRenderPy. "
            "Requires CUDA toolkit and Linux."
        )

    return info


# Surface materials for environmental rendering
TERRAIN_SURFACES = {
    "rough": lambda gs: gs.surfaces.Rough(color=(0.45, 0.38, 0.28)),
    "default": lambda gs, **kw: gs.surfaces.Default(**kw),
}

WATER_SURFACES = {
    "water": lambda gs: gs.surfaces.Water() if hasattr(gs.surfaces, "Water") else None,
    "default_water": lambda gs: gs.surfaces.Default(
        color=(0.2, 0.5, 0.9, 0.7), vis_mode="particle"
    ),
}
