"""Atmospheric data fetching from ERA5/GFS via Earth2Studio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326

DEFAULT_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "tp"]


def _try_import_data_source(source: str):
    """Import Earth2Studio data source, returning None if unavailable."""
    try:
        if source == "arco":
            from earth2studio.data import ARCO

            return ARCO
        if source == "gfs":
            from earth2studio.data import GFS

            return GFS
        logger.warning("Unknown data source: %s", source)
        return None
    except ImportError:
        logger.warning("earth2studio not installed for source '%s'", source)
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
    """
    variables = variables or DEFAULT_VARIABLES
    west, south, east, north = bbox

    arco_cls = _try_import_data_source("arco")
    if arco_cls is not None:
        return _fetch_via_earth2studio(arco_cls, time, variables, west, south, east, north)

    return _generate_synthetic(variables, west, south, east, north)


def fetch_gfs(
    bbox: Bbox,
    time: datetime,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Fetch GFS operational analysis data for a region and time.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        time: Target datetime.
        variables: Variable names. Defaults to DEFAULT_VARIABLES.

    Returns:
        xr.Dataset with variables as data vars.
    """
    variables = variables or DEFAULT_VARIABLES
    west, south, east, north = bbox

    gfs_cls = _try_import_data_source("gfs")
    if gfs_cls is not None:
        return _fetch_via_earth2studio(gfs_cls, time, variables, west, south, east, north)

    return _generate_synthetic(variables, west, south, east, north)


def fetch_atmosphere(
    bbox: Bbox,
    time: datetime,
    source: str = "era5",
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Unified atmospheric data fetch — picks the best available source.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        time: Target datetime.
        source: 'era5' (ARCO reanalysis), 'gfs' (operational), or 'auto'.
        variables: Variable names.

    Returns:
        xr.Dataset with atmospheric variables.
    """
    if source == "gfs":
        return fetch_gfs(bbox, time, variables)
    if source == "era5":
        return fetch_era5(bbox, time, variables)

    # auto: try ERA5 first, fall back to GFS, then synthetic
    for src_name, fetcher in [("era5", fetch_era5), ("gfs", fetch_gfs)]:
        try:
            ds = fetcher(bbox, time, variables)
            if ds is not None and len(ds.data_vars) > 0:
                logger.info("Using %s data source", src_name)
                return ds
        except Exception:
            logger.debug("Source %s failed, trying next", src_name, exc_info=True)

    variables = variables or DEFAULT_VARIABLES
    return _generate_synthetic(variables, *bbox)


def _fetch_via_earth2studio(
    source_cls: type,
    time: datetime,
    variables: list[str],
    west: float,
    south: float,
    east: float,
    north: float,
) -> xr.Dataset:
    """Fetch data using an Earth2Studio data source."""
    logger.info("Fetching via Earth2Studio %s for %s", source_cls.__name__, time)
    source = source_cls(cache=True, verbose=False)
    da = source(time=time, variable=variables)

    # Crop to bounding box
    lat_min, lat_max = min(south, north), max(south, north)
    lon_min, lon_max = min(west, east), max(west, east)

    # ERA5 ARCO / GFS may use 0-360° longitude
    lons = da.coords["lon"].values
    if lons.max() > 180:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
        logger.debug("Converted lon to 0-360 range: [%.1f, %.1f]", lon_min, lon_max)

    # Handle both ascending and descending lat
    lats = da.coords["lat"].values
    if lats[0] > lats[-1]:  # descending
        da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Convert DataArray to Dataset
    ds = xr.Dataset()
    for var in variables:
        var_data = da.sel(variable=var)
        if "variable" in var_data.dims:
            var_data = var_data.squeeze("variable", drop=True)
        if "time" in var_data.dims:
            var_data = var_data.squeeze("time", drop=True)
        ds[var] = var_data

    logger.info(
        "Dataset: %d variables, shape per var: %s",
        len(variables),
        ds[variables[0]].shape,
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

    ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    rng = np.random.default_rng(42)
    shape = (len(lat), len(lon))

    defaults = {
        "u10m": lambda: rng.normal(3.0, 2.0, shape).astype(np.float32),
        "v10m": lambda: rng.normal(1.0, 1.5, shape).astype(np.float32),
        "t2m": lambda: rng.normal(288.0, 5.0, shape).astype(np.float32),
        "tcwv": lambda: rng.uniform(10.0, 40.0, shape).astype(np.float32),
        "tp": lambda: rng.exponential(2.0, shape).astype(np.float32),
    }

    for var in variables:
        gen = defaults.get(var, lambda: rng.normal(0, 1, shape).astype(np.float32))
        ds[var] = xr.DataArray(gen(), dims=["lat", "lon"])

    return ds
