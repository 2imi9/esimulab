"""DEM (Digital Elevation Model) fetching and reprojection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

if TYPE_CHECKING:
    from rasterio.transform import Affine

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326


@dataclass(frozen=True)
class DEMResult:
    """Result of a DEM fetch operation."""

    heightfield: np.ndarray  # float32, shape (rows, cols), NaN filled for nodata
    pixel_size: float  # meters per pixel in projected CRS
    crs: CRS  # projected CRS (UTM)
    transform: Affine  # rasterio affine transform
    bounds: Bbox  # original geographic bounds


def _estimate_utm_zone(bbox: Bbox) -> CRS:
    """Estimate the best UTM zone for a bounding box centroid."""
    west, south, east, north = bbox
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2

    zone_number = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    epsg = 32600 + zone_number if hemisphere == "north" else 32700 + zone_number
    return CRS.from_epsg(epsg)


def _reproject_to_utm(
    data: np.ndarray,
    src_crs: CRS,
    src_transform: Affine,
    dst_crs: CRS,
) -> tuple[np.ndarray, Affine, float]:
    """Reproject a raster array from geographic to projected CRS.

    Returns:
        Tuple of (reprojected_array, new_transform, pixel_size_meters).
    """
    dst_transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        data.shape[1],
        data.shape[0],
        left=src_transform.c,
        bottom=src_transform.f + src_transform.e * data.shape[0],
        right=src_transform.c + src_transform.a * data.shape[1],
        top=src_transform.f,
    )

    dst_array = np.empty((height, width), dtype=np.float32)
    reproject(
        source=data,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    pixel_size = abs(dst_transform.a)  # meters per pixel
    return dst_array, dst_transform, pixel_size


def fetch_dem(
    bbox: Bbox,
    dem_name: str = "glo_30",
    reproject_to_utm: bool = True,
) -> DEMResult:
    """Fetch a DEM for the given bounding box and optionally reproject to UTM.

    Args:
        bbox: (west, south, east, north) in EPSG:4326 degrees.
        dem_name: DEM source name for dem-stitcher. Default 'glo_30' (Copernicus 30m).
        reproject_to_utm: If True, reproject from geographic to best-fit UTM zone.

    Returns:
        DEMResult with heightfield array and metadata.
    """
    from dem_stitcher import stitch_dem

    logger.info("Fetching DEM '%s' for bbox %s", dem_name, bbox)
    bounds_list = list(bbox)  # dem_stitcher expects [west, south, east, north]

    data, profile = stitch_dem(
        bounds_list,
        dem_name=dem_name,
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )

    data = data.astype(np.float32)
    src_crs = CRS.from_user_input(profile["crs"])
    src_transform = profile["transform"]

    if reproject_to_utm:
        dst_crs = _estimate_utm_zone(bbox)
        logger.info("Reprojecting DEM from %s to %s", src_crs, dst_crs)
        data, transform, pixel_size = _reproject_to_utm(data, src_crs, src_transform, dst_crs)
        crs = dst_crs
    else:
        transform = src_transform
        pixel_size = abs(src_transform.a)  # degrees, not meters
        crs = src_crs

    logger.info("DEM shape: %s, pixel_size: %.2f m", data.shape, pixel_size)

    return DEMResult(
        heightfield=data,
        pixel_size=pixel_size,
        crs=crs,
        transform=transform,
        bounds=bbox,
    )
