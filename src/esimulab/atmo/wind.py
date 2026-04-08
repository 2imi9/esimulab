"""Wind field extraction and Genesis force field parameter computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindForcing:
    """Wind parameters for Genesis force fields."""

    direction: tuple[float, float, float]  # normalized (x, y, z) unit vector
    magnitude: float  # mean wind speed (m/s)
    turbulence_strength: float  # sub-grid turbulence intensity
    turbulence_frequency: int  # Perlin noise frequency

    @property
    def direction_2d(self) -> tuple[float, float]:
        """Return (u, v) direction components."""
        return (self.direction[0], self.direction[1])


def extract_wind_forcing(
    ds: xr.Dataset,
    u_var: str = "u10m",
    v_var: str = "v10m",
    turbulence_fraction: float = 0.2,
    turbulence_freq: int = 5,
) -> WindForcing:
    """Extract wind forcing parameters from atmospheric dataset.

    Computes regional mean wind direction and magnitude, plus turbulence
    parameters scaled to sub-grid variability.

    Args:
        ds: xr.Dataset with u and v wind component variables.
        u_var: Name of the u-wind (eastward) variable.
        v_var: Name of the v-wind (northward) variable.
        turbulence_fraction: Turbulence strength as fraction of mean wind speed.
        turbulence_freq: Perlin noise frequency for turbulence force field.

    Returns:
        WindForcing with direction, magnitude, and turbulence parameters.
    """
    u_mean = float(ds[u_var].mean().values)
    v_mean = float(ds[v_var].mean().values)

    magnitude = float(np.sqrt(u_mean**2 + v_mean**2))

    if magnitude > 1e-6:
        direction = (u_mean / magnitude, v_mean / magnitude, 0.0)
    else:
        direction = (1.0, 0.0, 0.0)
        magnitude = 0.0

    turb_strength = magnitude * turbulence_fraction

    logger.info(
        "Wind forcing: dir=(%.2f, %.2f), speed=%.1f m/s, turb=%.1f",
        direction[0], direction[1], magnitude, turb_strength,
    )

    return WindForcing(
        direction=direction,
        magnitude=magnitude,
        turbulence_strength=turb_strength,
        turbulence_frequency=turbulence_freq,
    )
