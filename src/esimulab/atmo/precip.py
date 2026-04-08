"""Precipitation rate extraction for SPH rain emitter configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrecipForcing:
    """Precipitation parameters for Genesis SPH rain emitter."""

    rate_mm_hr: float  # regional mean precipitation rate (mm/hr)
    terminal_velocity: float  # raindrop terminal velocity (m/s)
    droplet_size: float  # SPH emitter droplet size parameter


def extract_precip_rate(
    ds: xr.Dataset,
    precip_var: str = "tp",
    terminal_velocity: float = 9.0,
    base_droplet_size: float = 0.05,
) -> PrecipForcing:
    """Extract precipitation forcing parameters from atmospheric dataset.

    Args:
        ds: xr.Dataset with precipitation variable.
        precip_var: Name of the total precipitation variable.
        terminal_velocity: Terminal fall velocity for raindrops (m/s).
        base_droplet_size: Base SPH droplet size for moderate rain.

    Returns:
        PrecipForcing with rate and emitter parameters.
    """
    if precip_var not in ds:
        logger.warning("Precipitation variable '%s' not in dataset, using zero", precip_var)
        return PrecipForcing(
            rate_mm_hr=0.0,
            terminal_velocity=terminal_velocity,
            droplet_size=base_droplet_size,
        )

    rate = float(ds[precip_var].mean().values)
    rate = max(0.0, rate)

    # Scale droplet size with intensity: heavier rain → larger droplets
    droplet_size = base_droplet_size * (1.0 + rate / 10.0)

    logger.info("Precipitation: %.2f mm/hr, droplet_size=%.3f", rate, droplet_size)

    return PrecipForcing(
        rate_mm_hr=rate,
        terminal_velocity=terminal_velocity,
        droplet_size=droplet_size,
    )
