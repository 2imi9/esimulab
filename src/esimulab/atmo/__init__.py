"""Atmospheric data acquisition and processing pipeline."""

from esimulab.atmo.fetch import fetch_atmosphere, fetch_era5, fetch_gfs
from esimulab.atmo.material_mapping import materials_from_atmosphere
from esimulab.atmo.precip import extract_precip_rate
from esimulab.atmo.wind import extract_wind_forcing

__all__ = [
    "extract_precip_rate",
    "extract_wind_forcing",
    "fetch_atmosphere",
    "fetch_era5",
    "fetch_gfs",
    "materials_from_atmosphere",
]
