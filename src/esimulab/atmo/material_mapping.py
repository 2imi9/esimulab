"""Map atmospheric conditions to Genesis material properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaterProperties:
    """Temperature-dependent water properties for SPH."""

    rho: float  # density (kg/m^3)
    mu: float  # dynamic viscosity (Pa*s)
    gamma: float  # surface tension coefficient
    stiffness: float  # EOS stiffness


@dataclass(frozen=True)
class EnvironmentalMaterials:
    """Complete material set derived from atmospheric conditions."""

    water: WaterProperties
    soil_type: str  # sand, elastic, liquid, snow
    description: str


def water_properties_from_temperature(temperature_k: float) -> WaterProperties:
    """Compute temperature-dependent water properties.

    Based on empirical fits for liquid water between 0-40°C.

    Args:
        temperature_k: Water temperature in Kelvin.

    Returns:
        WaterProperties with temperature-adjusted values.
    """
    t_c = temperature_k - 273.15  # Celsius
    t_c = max(0.0, min(40.0, t_c))  # clamp to valid range

    # Density: slight decrease with temperature
    # ρ(T) ≈ 1000 - 0.0178 * |T-4|^1.7 (simplified)
    rho = 1000.0 - 0.0178 * abs(t_c - 4.0) ** 1.7

    # Dynamic viscosity: decreases significantly with temperature
    # μ(T) ≈ 0.001 * exp(-0.02 * T) (simplified Arrhenius-like)
    import math

    mu = 0.00179 * math.exp(-0.0266 * t_c)  # Pa*s at T°C

    # Surface tension: slight decrease with temperature
    gamma = 0.0756 - 0.00014 * t_c

    # EOS stiffness (keep roughly constant for SPH stability)
    stiffness = 50000.0

    return WaterProperties(rho=rho, mu=mu, gamma=gamma, stiffness=stiffness)


def materials_from_atmosphere(
    temperature_k: float,
    humidity_kgm2: float = 20.0,
    precipitation_mm_hr: float = 0.0,
) -> EnvironmentalMaterials:
    """Derive Genesis material properties from atmospheric conditions.

    Args:
        temperature_k: 2m air temperature in Kelvin.
        humidity_kgm2: Total column water vapor (kg/m²).
        precipitation_mm_hr: Precipitation rate (mm/hr).

    Returns:
        EnvironmentalMaterials with water properties and soil type.
    """
    water = water_properties_from_temperature(temperature_k)

    # Soil type from temperature + moisture
    t_c = temperature_k - 273.15
    if t_c < -10:
        soil_type = "snow"
        desc = f"Frozen ({t_c:.0f}°C): snow cover, ice water"
    elif t_c < 0:
        soil_type = "elastic"
        desc = f"Cold ({t_c:.0f}°C): frozen ground, cold water (μ={water.mu:.4f})"
    elif precipitation_mm_hr > 10 or humidity_kgm2 > 45:
        soil_type = "liquid"
        desc = f"Saturated ({t_c:.0f}°C): wet soil, heavy rain"
    elif t_c > 30 and humidity_kgm2 < 15:
        soil_type = "sand"
        desc = f"Hot/dry ({t_c:.0f}°C): loose sand, warm water (μ={water.mu:.4f})"
    else:
        soil_type = "sand"
        desc = f"Moderate ({t_c:.0f}°C): standard soil (μ={water.mu:.4f})"

    logger.info("Material mapping: %s", desc)

    return EnvironmentalMaterials(water=water, soil_type=soil_type, description=desc)
