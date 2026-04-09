"""Spatially-varying wind zones from atmospheric grid data.

Partitions the simulation domain into zones, each with its own
Wind force field, to represent spatially-varying wind from ERA5/GFS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from esimulab.terrain.convert import GenesisHeightfield

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindZone:
    """A single wind zone with position and velocity."""

    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    strength: float
    radius: float


def create_wind_zones(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    heightfield: GenesisHeightfield,
    max_zones: int = 9,
) -> list[WindZone]:
    """Create spatially-varying wind zones from 2D wind fields.

    Partitions the terrain into a grid of zones, each with
    wind direction and magnitude from the nearest atmospheric data.

    Args:
        wind_u: 2D eastward wind component (m/s).
        wind_v: 2D northward wind component (m/s).
        heightfield: Terrain for spatial extent.
        max_zones: Maximum number of zones (will use sqrt x sqrt grid).

    Returns:
        List of WindZone instances.
    """
    bmin = heightfield.bounds_min
    bmax = heightfield.bounds_max

    n_side = int(np.sqrt(max_zones))
    n_side = max(1, min(n_side, min(wind_u.shape)))

    x_edges = np.linspace(bmin[0], bmax[0], n_side + 1)
    y_edges = np.linspace(bmin[1], bmax[1], n_side + 1)
    z_mid = (bmin[2] + bmax[2]) / 2

    # Sample wind at grid centers
    u_rows = np.linspace(0, wind_u.shape[0] - 1, n_side).astype(int)
    v_cols = np.linspace(0, wind_u.shape[1] - 1, n_side).astype(int)

    zones = []
    for i in range(n_side):
        for j in range(n_side):
            cx = (x_edges[j] + x_edges[j + 1]) / 2
            cy = (y_edges[i] + y_edges[i + 1]) / 2
            radius = max(x_edges[j + 1] - x_edges[j], y_edges[i + 1] - y_edges[i]) / 2

            u = float(wind_u[u_rows[i], v_cols[j]])
            v = float(wind_v[u_rows[i], v_cols[j]])
            mag = float(np.sqrt(u**2 + v**2))

            if mag < 0.01:
                continue

            direction = (u / mag, v / mag, 0.0)
            zones.append(
                WindZone(
                    center=(cx, cy, z_mid),
                    direction=direction,
                    strength=mag,
                    radius=radius,
                )
            )

    logger.info("Created %d wind zones from %dx%d grid", len(zones), n_side, n_side)
    return zones


def apply_wind_zones(gs: Any, scene: Any, zones: list[WindZone]) -> int:
    """Apply wind zones as Genesis force fields.

    Args:
        gs: Genesis module.
        scene: Genesis scene.
        zones: Wind zones to apply.

    Returns:
        Number of force fields added.
    """
    count = 0
    for zone in zones:
        try:
            scene.add_force_field(
                gs.engine.force_fields.Wind(
                    direction=zone.direction,
                    strength=zone.strength,
                    radius=zone.radius,
                    center=zone.center,
                )
            )
            count += 1
        except Exception:
            # Fall back to Constant if Wind type not available
            scene.add_force_field(
                gs.engine.force_fields.Constant(
                    direction=zone.direction,
                    strength=zone.strength,
                )
            )
            count += 1
            break  # Constant is global, only need one

    logger.info("Applied %d wind force fields", count)
    return count


def wind_zones_from_dataset(
    ds: xr.Dataset,
    heightfield: GenesisHeightfield,
    u_var: str = "u10m",
    v_var: str = "v10m",
    max_zones: int = 9,
) -> list[WindZone]:
    """Create wind zones directly from an xarray Dataset.

    Args:
        ds: Atmospheric dataset with wind components.
        heightfield: Terrain for spatial extent.
        u_var: Eastward wind variable name.
        v_var: Northward wind variable name.
        max_zones: Maximum zones.

    Returns:
        List of WindZone instances.
    """
    u = ds[u_var].values.squeeze()
    v = ds[v_var].values.squeeze()

    if u.ndim == 0:
        u = np.array([[float(u)]])
        v = np.array([[float(v)]])
    elif u.ndim == 1:
        u = u.reshape(1, -1)
        v = v.reshape(1, -1)

    return create_wind_zones(u, v, heightfield, max_zones)
