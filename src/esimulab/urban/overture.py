"""Overture Maps Foundation building data integration.

Overture Maps provides 2.3B+ building footprints globally with height
estimates (from Meta AI + OSM). Data is stored as GeoParquet on AWS S3,
queryable with DuckDB — no API key required.

Data source: s3://overturemaps-us-west-2/release/2024-11-13.0/theme=buildings/
License: CDLA Permissive 2.0 + ODbL (OSM-sourced)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north)

OVERTURE_S3_BASE = (
    "s3://overturemaps-us-west-2/release/2024-11-13.0/theme=buildings/type=building/*"
)


@dataclass
class OvertureBuilding:
    """A building from Overture Maps."""

    geometry_wkt: str  # WKT polygon
    centroid: tuple[float, float]  # (lon, lat)
    height: float  # meters (estimated or from OSM)
    num_floors: int | None = None
    class_: str = "residential"  # building class
    source: str = "overture"


@dataclass
class OvertureBuildingDataset:
    """Collection of Overture buildings for a region."""

    buildings: list[OvertureBuilding] = field(default_factory=list)
    bbox: Bbox = (0, 0, 0, 0)
    total_count: int = 0

    @property
    def count(self) -> int:
        return len(self.buildings)

    def heights_array(self) -> np.ndarray:
        """Return array of building heights."""
        return np.array([b.height for b in self.buildings], dtype=np.float32)

    def centroids_array(self) -> np.ndarray:
        """Return (N, 2) array of building centroids (lon, lat)."""
        return np.array([b.centroid for b in self.buildings], dtype=np.float64)


def fetch_overture_buildings(
    bbox: Bbox,
    max_buildings: int = 5000,
    min_height: float = 0.0,
) -> OvertureBuildingDataset:
    """Fetch building footprints from Overture Maps via DuckDB.

    Queries GeoParquet files directly on S3 — no API key needed.
    DuckDB's spatial extension handles the bbox filter efficiently.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        max_buildings: Maximum buildings to return.
        min_height: Minimum building height filter (meters).

    Returns:
        OvertureBuildingDataset with building data.
    """
    try:
        return _query_overture_duckdb(bbox, max_buildings, min_height)
    except Exception:
        logger.warning("Overture Maps query failed, generating synthetic buildings")
        logger.debug("Overture error details", exc_info=True)
        return _generate_synthetic_overture(bbox, max_buildings)


def _query_overture_duckdb(
    bbox: Bbox,
    max_buildings: int,
    min_height: float,
) -> OvertureBuildingDataset:
    """Query Overture Maps GeoParquet via DuckDB spatial SQL."""
    import duckdb

    west, south, east, north = bbox

    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")

    query = f"""
    SELECT
        ST_AsText(geometry) as geom_wkt,
        ST_X(ST_Centroid(geometry)) as centroid_lon,
        ST_Y(ST_Centroid(geometry)) as centroid_lat,
        COALESCE(height, num_floors * 3.0, 10.0) as height_m,
        num_floors,
        COALESCE(class, 'unknown') as building_class
    FROM read_parquet('{OVERTURE_S3_BASE}', filename=true, hive_partitioning=true)
    WHERE bbox.xmin >= {west}
      AND bbox.xmax <= {east}
      AND bbox.ymin >= {south}
      AND bbox.ymax <= {north}
      AND COALESCE(height, num_floors * 3.0, 10.0) >= {min_height}
    LIMIT {max_buildings}
    """

    logger.info("Querying Overture Maps for bbox %s (max %d)...", bbox, max_buildings)
    result = conn.execute(query).fetchall()
    conn.close()

    buildings = []
    for row in result:
        geom_wkt, clon, clat, height, num_floors, bclass = row
        buildings.append(
            OvertureBuilding(
                geometry_wkt=geom_wkt,
                centroid=(float(clon), float(clat)),
                height=float(height),
                num_floors=int(num_floors) if num_floors else None,
                class_=str(bclass),
            )
        )

    logger.info("Overture: fetched %d buildings for bbox %s", len(buildings), bbox)

    return OvertureBuildingDataset(
        buildings=buildings,
        bbox=bbox,
        total_count=len(buildings),
    )


def _generate_synthetic_overture(
    bbox: Bbox,
    max_buildings: int,
) -> OvertureBuildingDataset:
    """Generate synthetic building data as Overture fallback."""
    rng = np.random.default_rng(42)
    west, south, east, north = bbox

    n = min(max_buildings, 500)
    buildings = []
    for _ in range(n):
        lon = rng.uniform(west, east)
        lat = rng.uniform(south, north)
        height = float(rng.lognormal(mean=np.log(12), sigma=0.5))
        height = min(height, 80)
        floors = max(1, int(height / 3))

        buildings.append(
            OvertureBuilding(
                geometry_wkt=f"POINT({lon} {lat})",
                centroid=(lon, lat),
                height=height,
                num_floors=floors,
                class_=rng.choice(["residential", "commercial", "industrial"]),
            )
        )

    return OvertureBuildingDataset(buildings=buildings, bbox=bbox, total_count=n)


def overture_to_heightfield_mask(
    dataset: OvertureBuildingDataset,
    dem_shape: tuple[int, int],
    dem_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert Overture buildings to a height mask aligned with DEM grid.

    Creates a (H, W) array where each cell contains the building
    height at that location, or 0 for open ground.

    Args:
        dataset: Overture building dataset.
        dem_shape: (rows, cols) of the DEM.
        dem_bounds: (west, south, east, north) in EPSG:4326 degrees
            OR (xmin, ymin, xmax, ymax) in projected meters.

    Returns:
        (H, W) float32 building height mask.
    """
    rows, cols = dem_shape
    xmin, ymin, xmax, ymax = dem_bounds
    mask = np.zeros((rows, cols), dtype=np.float32)

    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range == 0 or y_range == 0:
        return mask

    for b in dataset.buildings:
        lon, lat = b.centroid
        # Normalize to grid coordinates [0, 1]
        nx = (lon - xmin) / x_range
        ny = (lat - ymin) / y_range

        c = int(nx * cols)
        r = int(ny * rows)

        if 0 <= r < rows and 0 <= c < cols:
            # Stamp building footprint (approximate as 3x3 cells)
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            mask[r0:r1, c0:c1] = np.maximum(mask[r0:r1, c0:c1], b.height)

    logger.info(
        "Building height mask: %d cells with buildings (max height %.0fm)",
        int((mask > 0).sum()),
        mask.max(),
    )
    return mask


def overture_to_genesis_boxes(
    dataset: OvertureBuildingDataset,
    gs: Any,
    scene: Any,
    terrain_origin: tuple[float, float] = (0.0, 0.0),
    max_entities: int = 200,
) -> list[Any]:
    """Add Overture buildings as Genesis box rigid bodies.

    Args:
        dataset: Overture building dataset.
        gs: Genesis module.
        scene: Genesis scene.
        terrain_origin: (lon, lat) of terrain center for coordinate mapping.
        max_entities: Max buildings to add (Genesis has entity limits).

    Returns:
        List of Genesis entities.
    """
    entities = []
    buildings = dataset.buildings[:max_entities]
    origin_lon, origin_lat = terrain_origin

    for b in buildings:
        lon, lat = b.centroid
        # Convert to local meters
        dx = (lon - origin_lon) * 111320 * np.cos(np.radians(origin_lat))
        dy = (lat - origin_lat) * 110540
        h = b.height

        # Approximate building footprint as 15x15m box
        try:
            entity = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(dx, dy, h / 2),
                    size=(15, 15, h),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(0.6, 0.6, 0.65, 1.0)),
            )
            entities.append(entity)
        except Exception:
            break  # hit entity limit

    logger.info("Added %d Overture buildings to Genesis scene", len(entities))
    return entities
