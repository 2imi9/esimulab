"""Terrain data acquisition and processing pipeline."""

from esimulab.terrain.convert import prepare_heightfield
from esimulab.terrain.dem import fetch_dem
from esimulab.terrain.landcover import fetch_landcover

__all__ = ["fetch_dem", "fetch_landcover", "prepare_heightfield"]
