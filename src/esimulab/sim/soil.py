"""MPM soil and sediment configuration for Genesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from esimulab.terrain.convert import GenesisHeightfield

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SoilConfig:
    """Configuration for MPM soil layer."""

    material_type: str = "sand"  # sand, elastic, liquid, snow
    thickness: float = 2.0  # meters
    grid_density: int = 64
    coverage_fraction: float = 0.8  # fraction of terrain covered


# Mapping from land cover classes to soil material types
LANDCOVER_TO_SOIL = {
    10: "elastic",  # tree cover → compacted earth
    20: "sand",  # shrubland → sandy soil
    30: "sand",  # grassland → sandy soil
    40: "elastic",  # cropland → compacted earth
    50: "elastic",  # built-up → compacted
    60: "sand",  # bare/sparse → loose sand
    70: "snow",  # snow/ice → snow material
    80: None,  # permanent water → no soil
    90: "liquid",  # wetland → saturated soil
    95: "liquid",  # mangroves → saturated
    100: "elastic",  # moss/lichen → compacted
}


def _get_mpm_material(gs: Any, material_type: str) -> Any:
    """Get the appropriate Genesis MPM material."""
    materials = {
        "sand": lambda: gs.materials.MPM.Sand(),
        "elastic": lambda: gs.materials.MPM.Elastic(),
        "liquid": lambda: gs.materials.MPM.Liquid(),
        "snow": lambda: gs.materials.MPM.Snow(),
    }
    factory = materials.get(material_type)
    if factory is None:
        logger.warning("Unknown soil type '%s', defaulting to Sand", material_type)
        return gs.materials.MPM.Sand()
    return factory()


def add_soil_layer(
    gs: Any,
    scene: Any,
    heightfield: GenesisHeightfield,
    config: SoilConfig | None = None,
) -> Any | None:
    """Add an MPM soil layer to the Genesis scene.

    Creates a thin box of MPM material covering the terrain surface.

    Args:
        gs: Genesis module.
        scene: Genesis scene.
        heightfield: Terrain heightfield data.
        config: Soil configuration. Defaults to sand.

    Returns:
        The soil entity, or None if soil is disabled.
    """
    config = config or SoilConfig()

    material = _get_mpm_material(gs, config.material_type)

    # Soil layer positioned just above the terrain minimum
    bmin = heightfield.bounds_min
    bmax = heightfield.bounds_max
    center_x = (bmin[0] + bmax[0]) / 2
    center_y = (bmin[1] + bmax[1]) / 2
    z_base = bmin[2] + (bmax[2] - bmin[2]) * 0.3  # lower third of terrain

    extent_x = (bmax[0] - bmin[0]) * config.coverage_fraction
    extent_y = (bmax[1] - bmin[1]) * config.coverage_fraction

    soil_colors = {
        "sand": (0.76, 0.60, 0.42),
        "elastic": (0.55, 0.40, 0.26),
        "liquid": (0.40, 0.32, 0.22),
        "snow": (0.95, 0.95, 0.97),
    }
    color = soil_colors.get(config.material_type, (0.6, 0.4, 0.2))

    soil = scene.add_entity(
        material=material,
        morph=gs.morphs.Box(
            pos=(center_x, center_y, z_base),
            size=(extent_x, extent_y, config.thickness),
        ),
        surface=gs.surfaces.Default(
            color=(*color, 1.0),
            vis_mode="particle",
        ),
    )

    logger.info(
        "Soil layer added: type=%s, thickness=%.1fm, extent=%.0fx%.0fm",
        config.material_type,
        config.thickness,
        extent_x,
        extent_y,
    )

    return soil


def soil_config_from_temperature(temperature_k: float) -> SoilConfig:
    """Select soil material based on temperature.

    Args:
        temperature_k: Temperature in Kelvin.

    Returns:
        SoilConfig with appropriate material type.
    """
    if temperature_k < 263:  # -10°C
        return SoilConfig(material_type="snow", thickness=1.0)
    if temperature_k < 273:  # 0°C
        return SoilConfig(material_type="elastic", thickness=2.0)  # frozen ground
    if temperature_k < 288:  # 15°C
        return SoilConfig(material_type="sand", thickness=2.0)
    return SoilConfig(material_type="sand", thickness=2.5)  # warm, drier soil
