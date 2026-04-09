"""Urban wind canyon effect modeling.

Buildings create urban canyons that:
- Accelerate wind through narrow passages (Venturi effect)
- Create turbulent wakes behind buildings
- Redirect wind along street corridors
- Generate downdrafts at building faces

This module modifies Genesis force fields based on building geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanyonEffect:
    """Wind modification at a specific location due to urban canyon."""

    speedup_factor: float  # 1.0=no effect, >1=acceleration
    direction_deflection: float  # degrees of wind direction change
    turbulence_intensity: float  # additional turbulence (m/s)
    downdraft_strength: float  # vertical wind component (m/s, negative=down)


def compute_canyon_speedup(
    building_heights: np.ndarray,
    wind_direction: tuple[float, float],
    pixel_size: float,
) -> np.ndarray:
    """Compute wind speed modification factor from building geometry.

    Uses a simplified canyon ratio model:
    - Canyon ratio = H/W (building height / street width)
    - Speedup occurs at canyon entrances and above-roof level
    - Deceleration occurs in wide canyons (sheltering)

    Args:
        building_heights: (H, W) array, 0 for open, >0 for buildings.
        wind_direction: (u, v) normalized wind direction.
        pixel_size: Grid spacing in meters.

    Returns:
        (H, W) float32 speedup factor (1.0 = no change).
    """
    h, w = building_heights.shape
    speedup = np.ones((h, w), dtype=np.float32)

    # Find building edges (canyon entrances)
    is_building = building_heights > 0
    building_height = building_heights.copy()
    building_height[~is_building] = 0

    # Compute average building height in neighborhood
    from scipy.ndimage import uniform_filter

    avg_height = uniform_filter(building_height, size=5)

    # Canyon ratio approximation
    # Street width ≈ distance to nearest building in wind direction
    # Simplified: use gap between buildings
    gap = np.ones((h, w), dtype=np.float32) * pixel_size * 5  # default 5 cells wide
    gap[is_building] = 0

    safe_gap = np.where(gap > 0, gap, 1.0)
    canyon_ratio = np.where(gap > 0, avg_height / safe_gap, 0)

    # Speedup at canyon entrances (H/W > 0.5 → acceleration)
    # Based on Oke (1988) street canyon classification:
    # H/W < 0.3: isolated roughness (little effect)
    # 0.3 < H/W < 0.7: wake interference
    # H/W > 0.7: skimming flow (strong channeling)
    speedup_factor = np.where(
        canyon_ratio > 0.7, 1.3 + 0.2 * np.minimum(canyon_ratio - 0.7, 1.0),
        np.where(canyon_ratio > 0.3, 1.0 + 0.3 * (canyon_ratio - 0.3) / 0.4, 1.0),
    )

    # Deceleration in building shadows (wake)
    # Buildings block wind: deceleration behind buildings
    wind_u, wind_v = wind_direction
    if abs(wind_u) > abs(wind_v):
        # Predominantly east-west wind
        shift = int(np.sign(wind_u))
        wake = np.roll(is_building, -shift * 3, axis=1)  # 3 cells downwind
        speedup[wake & ~is_building] *= 0.6
    else:
        shift = int(np.sign(wind_v))
        wake = np.roll(is_building, -shift * 3, axis=0)
        speedup[wake & ~is_building] *= 0.6

    speedup[~is_building] *= speedup_factor[~is_building]
    speedup[is_building] = 0.0  # no wind inside buildings

    logger.info(
        "Canyon speedup: mean=%.2f, max=%.2f, buildings=%d cells",
        speedup[~is_building].mean() if (~is_building).any() else 0,
        speedup.max(),
        is_building.sum(),
    )
    return speedup


def compute_urban_turbulence(
    building_heights: np.ndarray,
    wind_speed: float,
    pixel_size: float,
) -> np.ndarray:
    """Compute additional turbulence intensity from buildings.

    Buildings generate turbulent kinetic energy through:
    - Form drag on building faces
    - Wake turbulence behind buildings
    - Rooftop shear layers

    Args:
        building_heights: (H, W) building height array.
        wind_speed: Ambient wind speed (m/s).
        pixel_size: Grid spacing (m).

    Returns:
        (H, W) float32 additional turbulence intensity (m/s).
    """
    is_building = building_heights > 0

    # Turbulence proportional to building height and wind speed
    # TKE ∝ u²·Cd where Cd is drag coefficient (~1.2 for bluff bodies)
    cd = 1.2
    base_turbulence = 0.5 * cd * (wind_speed / 10) ** 2  # normalized

    # Higher turbulence near building edges
    from scipy.ndimage import binary_dilation

    near_building = binary_dilation(is_building, iterations=2) & ~is_building
    turbulence = np.zeros_like(building_heights, dtype=np.float32)
    turbulence[near_building] = base_turbulence * building_heights.max() / 20
    turbulence[is_building] = 0  # inside buildings

    return turbulence


def create_urban_wind_zones(
    building_heights: np.ndarray,
    base_wind_dir: tuple[float, float, float],
    base_wind_speed: float,
    pixel_size: float,
    heightfield_bounds: tuple[tuple, tuple],
) -> list[dict]:
    """Create modified wind zones accounting for urban canyon effects.

    Args:
        building_heights: (H, W) building heights (0=open).
        base_wind_dir: Base wind direction (x, y, z).
        base_wind_speed: Base wind speed (m/s).
        pixel_size: Grid spacing (m).
        heightfield_bounds: ((xmin,ymin,zmin), (xmax,ymax,zmax)).

    Returns:
        List of wind zone dicts with modified direction and strength.
    """
    speedup = compute_canyon_speedup(
        building_heights, (base_wind_dir[0], base_wind_dir[1]), pixel_size
    )
    turbulence = compute_urban_turbulence(building_heights, base_wind_speed, pixel_size)

    bmin, bmax = heightfield_bounds
    h, w = building_heights.shape

    zones = []
    # Create 3x3 grid of zones with canyon-modified wind
    n_zones = 3
    for i in range(n_zones):
        for j in range(n_zones):
            r = int((i + 0.5) * h / n_zones)
            c = int((j + 0.5) * w / n_zones)
            r = min(r, h - 1)
            c = min(c, w - 1)

            local_speedup = float(speedup[r, c])
            local_turb = float(turbulence[r, c])
            local_speed = base_wind_speed * local_speedup

            cx = bmin[0] + (j + 0.5) * (bmax[0] - bmin[0]) / n_zones
            cy = bmin[1] + (i + 0.5) * (bmax[1] - bmin[1]) / n_zones
            cz = (bmin[2] + bmax[2]) / 2

            zones.append({
                "center": (cx, cy, cz),
                "direction": base_wind_dir,
                "strength": local_speed,
                "turbulence": local_turb,
                "speedup": local_speedup,
            })

    logger.info("Created %d urban wind zones", len(zones))
    return zones
