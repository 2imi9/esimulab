"""ESA WorldCover land cover data fetching and alignment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

# ESA WorldCover 2021 class mapping
LANDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# WorldCover S3 bucket and tile structure
_WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
_WORLDCOVER_VERSION = "v200"


def _tile_urls_for_bbox(
    west: float, south: float, east: float, north: float,
) -> list[str]:
    """Generate WorldCover 3x3 degree tile URLs covering a bounding box."""
    import math

    urls = []
    lat_start = math.floor(south / 3) * 3
    lat_end = math.ceil(north / 3) * 3
    lon_start = math.floor(west / 3) * 3
    lon_end = math.ceil(east / 3) * 3

    for lat in range(lat_start, lat_end, 3):
        for lon in range(lon_start, lon_end, 3):
            lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tile_name = f"ESA_WorldCover_10m_2021_{_WORLDCOVER_VERSION}_{lat_str}{lon_str}_Map"
            url = f"{_WORLDCOVER_BASE}/{_WORLDCOVER_VERSION}/2021/map/{tile_name}/{tile_name}.tif"
            urls.append(url)

    return urls


def fetch_landcover(
    bbox: tuple[float, float, float, float],
    dem_profile: dict[str, Any],
) -> np.ndarray:
    """Fetch ESA WorldCover and align to DEM grid.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        dem_profile: Dict with 'crs', 'transform', 'width', 'height' from the DEM
            (as returned by rasterio or DEMResult metadata).

    Returns:
        uint8 array aligned to DEM grid with ESA WorldCover class values.
    """
    urls = _tile_urls_for_bbox(*bbox)
    logger.info("Fetching %d WorldCover tile(s)", len(urls))

    tiles = []
    for url in urls:
        try:
            tile = rioxarray.open_rasterio(url, chunks="auto")
            tiles.append(tile)
        except Exception:
            logger.warning("Could not fetch tile: %s", url)
            continue

    if not tiles:
        logger.warning("No WorldCover tiles available, returning zeros")
        height = dem_profile.get("height", 100)
        width = dem_profile.get("width", 100)
        return np.zeros((height, width), dtype=np.uint8)

    mosaic = tiles[0] if len(tiles) == 1 else xr.concat(tiles, dim="x")

    # Crop to bbox
    west, south, east, north = bbox
    mosaic = mosaic.rio.clip_box(minx=west, miny=south, maxx=east, maxy=north)

    # Build a target DataArray matching the DEM grid for reprojection
    dst_crs = CRS.from_user_input(dem_profile["crs"])
    transform = dem_profile["transform"]
    height = dem_profile.get("height", mosaic.shape[-2])
    width = dem_profile.get("width", mosaic.shape[-1])

    aligned = mosaic.rio.reproject(
        dst_crs,
        shape=(height, width),
        transform=transform,
        resampling=0,  # nearest neighbor for categorical data
    )

    result = aligned.values.squeeze().astype(np.uint8)
    logger.info("Land cover shape: %s, unique classes: %s", result.shape, np.unique(result))
    return result
