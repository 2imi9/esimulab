"""Terrain data acquisition and processing pipeline."""

from esimulab.terrain.convert import prepare_heightfield
from esimulab.terrain.dem import fetch_dem

__all__ = ["fetch_dem", "prepare_heightfield"]
