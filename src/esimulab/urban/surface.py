"""Impervious surface modeling and urban hydrology effects.

Urban surfaces dramatically alter water runoff:
- Natural soil: infiltrates 40-70% of rainfall
- Asphalt/concrete: 90-95% becomes surface runoff
- Rooftops: 85-95% runoff
- Urban parks: 10-25% runoff (compacted soil)

This module computes spatially-varying runoff coefficients from
land cover data and building footprints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Runoff coefficients by surface type (fraction of rainfall that becomes runoff)
SURFACE_RUNOFF_COEFFICIENTS = {
    "asphalt": 0.90,
    "concrete": 0.92,
    "rooftop": 0.88,
    "gravel": 0.50,
    "compacted_soil": 0.45,
    "lawn": 0.20,
    "forest": 0.10,
    "bare_soil": 0.35,
    "water": 1.00,  # already water → full "runoff"
    "wetland": 0.15,
}

# ESA WorldCover class → imperviousness fraction
LANDCOVER_IMPERVIOUSNESS = {
    10: 0.02,   # tree cover → near-zero impervious
    20: 0.05,   # shrubland
    30: 0.05,   # grassland
    40: 0.10,   # cropland (compacted rows)
    50: 0.75,   # built-up → high imperviousness
    60: 0.15,   # bare/sparse
    70: 0.30,   # snow/ice (frozen = somewhat impervious)
    80: 1.00,   # permanent water
    90: 0.05,   # herbaceous wetland
    95: 0.05,   # mangroves
    100: 0.03,  # moss/lichen
}


@dataclass(frozen=True)
class UrbanSurfaceProperties:
    """Urban surface properties for a grid cell."""

    impervious_fraction: float  # 0-1, fraction of impervious surface
    runoff_coefficient: float  # 0-1, fraction of rain → runoff
    infiltration_rate: float  # mm/hr, soil infiltration capacity
    heat_capacity_factor: float  # multiplier for UHI effect


def compute_impervious_fraction(
    landcover: np.ndarray,
    building_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spatially-varying impervious surface fraction.

    Combines ESA WorldCover land cover classes with optional
    building footprint mask for higher accuracy.

    Args:
        landcover: (H, W) uint8 ESA WorldCover classes.
        building_mask: Optional (H, W) binary mask (1=building).

    Returns:
        (H, W) float32 array with imperviousness 0-1.
    """
    imperviousness = np.zeros_like(landcover, dtype=np.float32)

    for lc_class, fraction in LANDCOVER_IMPERVIOUSNESS.items():
        imperviousness[landcover == lc_class] = fraction

    # Override with building mask if available (buildings → 0.95 impervious)
    if building_mask is not None:
        imperviousness[building_mask > 0] = np.maximum(
            imperviousness[building_mask > 0], 0.95
        )

    logger.info(
        "Impervious fraction: mean=%.2f, max=%.2f, built-up cells=%d",
        imperviousness.mean(),
        imperviousness.max(),
        int((imperviousness > 0.5).sum()),
    )
    return imperviousness


def urban_runoff_coefficient(
    landcover: np.ndarray,
    building_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spatially-varying runoff coefficients.

    Higher values mean more rainfall becomes surface runoff
    (less infiltration into soil).

    Args:
        landcover: (H, W) uint8 ESA WorldCover classes.
        building_mask: Optional (H, W) binary mask.

    Returns:
        (H, W) float32 runoff coefficient 0-1.
    """
    imperviousness = compute_impervious_fraction(landcover, building_mask)

    # Weighted blend: impervious surfaces have high runoff, pervious low
    # C = C_impervious * f_imp + C_pervious * (1 - f_imp)
    c_impervious = 0.90
    c_pervious = 0.20
    runoff_coeff = c_impervious * imperviousness + c_pervious * (1 - imperviousness)

    return runoff_coeff.astype(np.float32)


def urban_infiltration_rate(
    landcover: np.ndarray,
    building_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute spatially-varying soil infiltration rate.

    Infiltration is the rate at which water enters the soil.
    Impervious surfaces have near-zero infiltration.

    Args:
        landcover: (H, W) uint8 ESA WorldCover classes.
        building_mask: Optional (H, W) binary mask.

    Returns:
        (H, W) float32 infiltration rate in mm/hr.
    """
    # Base infiltration rates by land cover (mm/hr)
    base_rates = {
        10: 25.0,   # forest → high infiltration
        20: 15.0,   # shrubland
        30: 12.0,   # grassland
        40: 10.0,   # cropland
        50: 2.0,    # built-up → low
        60: 8.0,    # bare
        70: 0.5,    # snow/ice → very low
        80: 0.0,    # water → zero
        90: 20.0,   # wetland → high (saturated though)
        95: 18.0,   # mangroves
        100: 15.0,  # moss/lichen
    }

    infiltration = np.zeros_like(landcover, dtype=np.float32)
    for lc_class, rate in base_rates.items():
        infiltration[landcover == lc_class] = rate

    # Buildings have near-zero infiltration
    if building_mask is not None:
        infiltration[building_mask > 0] = 0.5

    return infiltration


def urban_heat_island_adjustment(
    temperature_k: float,
    impervious_fraction: np.ndarray,
    time_of_day_hour: int = 14,
) -> np.ndarray:
    """Compute Urban Heat Island temperature adjustment.

    UHI effect is strongest:
    - At night (3-5°C)
    - In highly urbanized areas (impervious > 0.7)
    - During clear, calm conditions

    Args:
        temperature_k: Base 2m temperature in Kelvin.
        impervious_fraction: (H, W) imperviousness 0-1.
        time_of_day_hour: Hour (0-23) for diurnal variation.

    Returns:
        (H, W) float32 temperature adjustment in Kelvin.
    """
    # UHI intensity scales with imperviousness
    # Max UHI ≈ 5°C for fully urbanized areas
    max_uhi = 5.0
    uhi_base = impervious_fraction * max_uhi

    # Diurnal variation: stronger at night (18:00-06:00)
    # Peak at ~02:00 (hour=2), minimum at ~14:00 (hour=14)
    import math

    hour_angle = (time_of_day_hour - 2) / 24 * 2 * math.pi
    diurnal_factor = 0.5 + 0.5 * math.cos(hour_angle)  # 1.0 at night, 0.0 at day

    # Daytime UHI is about 40% of nighttime
    diurnal_factor = 0.4 + 0.6 * diurnal_factor

    uhi_adjustment = uhi_base * diurnal_factor

    logger.info(
        "UHI adjustment: mean=+%.1f°C, max=+%.1f°C (hour=%d)",
        uhi_adjustment.mean(), uhi_adjustment.max(), time_of_day_hour,
    )

    return uhi_adjustment.astype(np.float32)
