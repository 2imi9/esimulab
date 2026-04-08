"""Atmospheric data fetching from ERA5 via Earth2Studio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime
import xarray as xr

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326

DEFAULT_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "tp"]


def _try_import_earth2studio():
    """Import Earth2Studio ARCO data source, returning None if unavailable."""
    try:
        from earth2studio.data import ARCO

        return ARCO
    except ImportError:
        logger.warning(
            "earth2studio not installed. Install with: pip install earth2studio"
        )
        return None


def fetch_era5(
    bbox: Bbox,
    time: datetime,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Fetch ERA5 reanalysis data for a region and time.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        time: Target datetime for the atmospheric snapshot.
        variables: List of ERA5 variable names. Defaults to
            ['u10m', 'v10m', 't2m', 'tcwv', 'tp'].

    Returns:
        xr.Dataset with variables as data vars, dimensions [lat, lon].

    Raises:
        ImportError: If earth2studio is not installed.
        RuntimeError: If data fetch fails.
    """
    variables = variables or DEFAULT_VARIABLES
    west, south, east, north = bbox

    arco_cls = _try_import_earth2studio()

    if arco_cls is not None:
        return _fetch_via_earth2studio(arco_cls, time, variables, west, south, east, north)

    return _generate_synthetic(variables, west, south, east, north)


def _fetch_via_earth2studio(
    arco_cls: type,
    time: datetime,
    variables: list[str],
    west: float,
    south: float,
    east: float,
    north: float,
) -> xr.Dataset:
    """Fetch data using Earth2Studio ARCO data source."""
    logger.info("Fetching ERA5 via Earth2Studio ARCO for %s", time)
    arco = arco_cls(cache=True, verbose=False)
    da = arco(time=time, variable=variables)

    # Crop to bounding box
    # ERA5 lat is typically descending (90 to -90)
    lat_min, lat_max = min(south, north), max(south, north)
    lon_min, lon_max = min(west, east), max(west, east)

    # Handle both ascending and descending lat
    lats = da.coords["lat"].values
    if lats[0] > lats[-1]:  # descending
        da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Convert DataArray to Dataset with each variable as a data var
    ds = xr.Dataset()
    for var in variables:
        var_data = da.sel(variable=var)
        if "variable" in var_data.dims:
            var_data = var_data.squeeze("variable", drop=True)
        if "time" in var_data.dims:
            var_data = var_data.squeeze("time", drop=True)
        ds[var] = var_data

    logger.info(
        "ERA5 dataset: %d variables, shape per var: %s",
        len(variables), ds[variables[0]].shape,
    )
    return ds


def _generate_synthetic(
    variables: list[str],
    west: float,
    south: float,
    east: float,
    north: float,
) -> xr.Dataset:
    """Generate synthetic atmospheric data for testing without Earth2Studio."""
    logger.warning("Using synthetic atmospheric data (Earth2Studio not available)")

    lat = np.arange(south, north, 0.25)
    lon = np.arange(west, east, 0.25)

    ds = xr.Dataset(
        coords={"lat": lat, "lon": lon},
    )

    rng = np.random.default_rng(42)
    shape = (len(lat), len(lon))

    defaults = {
        "u10m": lambda: rng.normal(3.0, 2.0, shape).astype(np.float32),
        "v10m": lambda: rng.normal(1.0, 1.5, shape).astype(np.float32),
        "t2m": lambda: (rng.normal(288.0, 5.0, shape)).astype(np.float32),
        "tcwv": lambda: rng.uniform(10.0, 40.0, shape).astype(np.float32),
        "tp": lambda: rng.exponential(2.0, shape).astype(np.float32),
    }

    for var in variables:
        gen = defaults.get(var, lambda: rng.normal(0, 1, shape).astype(np.float32))
        ds[var] = xr.DataArray(gen(), dims=["lat", "lon"])

    return ds
