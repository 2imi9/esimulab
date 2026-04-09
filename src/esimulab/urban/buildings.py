"""Building footprint acquisition and 3D mesh extrusion.

Data sources:
- OpenStreetMap (Overpass API) — global, community-maintained
- Microsoft Building Footprints — ML-extracted, high coverage
- ESA WorldCover class 50 — 10m built-up classification

Buildings are extruded to 3D meshes and loaded as Genesis rigid
bodies for wind channeling and rainfall interception simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north)


@dataclass
class Building:
    """A single building with footprint and height."""

    footprint: np.ndarray  # (N, 2) polygon vertices in UTM meters
    height: float  # meters
    ground_elevation: float = 0.0  # base elevation from DEM
    building_type: str = "residential"  # residential, commercial, industrial


@dataclass
class BuildingDataset:
    """Collection of buildings for a region."""

    buildings: list[Building] = field(default_factory=list)
    bbox: Bbox = (0, 0, 0, 0)
    source: str = "unknown"
    crs: str = "EPSG:4326"

    @property
    def count(self) -> int:
        return len(self.buildings)

    @property
    def total_footprint_area(self) -> float:
        """Total building footprint area in m²."""
        total = 0.0
        for b in self.buildings:
            if len(b.footprint) >= 3:
                # Shoelace formula
                x = b.footprint[:, 0]
                y = b.footprint[:, 1]
                total += abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 2)
        return total


def fetch_building_footprints(
    bbox: Bbox,
    source: str = "osm",
    default_height: float = 10.0,
) -> BuildingDataset:
    """Fetch building footprints for a bounding box.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        source: 'osm' (OpenStreetMap), 'microsoft', or 'synthetic'.
        default_height: Default building height when not available.

    Returns:
        BuildingDataset with building footprints.
    """
    if source == "osm":
        return _fetch_osm_buildings(bbox, default_height)
    if source == "microsoft":
        return _fetch_microsoft_buildings(bbox, default_height)
    return _generate_synthetic_buildings(bbox, default_height)


def _fetch_osm_buildings(
    bbox: Bbox,
    default_height: float,
) -> BuildingDataset:
    """Fetch buildings from OpenStreetMap via Overpass API."""
    try:
        import requests

        west, south, east, north = bbox
        query = f"""
        [out:json][timeout:30];
        way["building"]({south},{west},{north},{east});
        out body geom;
        """

        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        buildings = []
        for element in data.get("elements", []):
            if "geometry" not in element:
                continue

            coords = np.array(
                [(pt["lon"], pt["lat"]) for pt in element["geometry"]],
                dtype=np.float64,
            )

            # Extract height from tags
            tags = element.get("tags", {})
            height = _parse_building_height(tags, default_height)
            btype = tags.get("building", "residential")

            buildings.append(Building(
                footprint=coords,
                height=height,
                building_type=btype,
            ))

        logger.info("OSM: fetched %d buildings for bbox %s", len(buildings), bbox)
        return BuildingDataset(buildings=buildings, bbox=bbox, source="osm")

    except Exception:
        logger.warning("OSM fetch failed, generating synthetic buildings")
        return _generate_synthetic_buildings(bbox, default_height)


def _parse_building_height(tags: dict, default: float) -> float:
    """Extract building height from OSM tags."""
    # Try explicit height tag
    height_str = tags.get("height", tags.get("building:height", ""))
    if height_str:
        try:
            return float(height_str.replace("m", "").strip())
        except ValueError:
            pass

    # Estimate from levels
    levels_str = tags.get("building:levels", "")
    if levels_str:
        try:
            return float(levels_str) * 3.0  # ~3m per floor
        except ValueError:
            pass

    return default


def _fetch_microsoft_buildings(
    bbox: Bbox,
    default_height: float,
) -> BuildingDataset:
    """Fetch from Microsoft Building Footprints (placeholder)."""
    logger.warning("Microsoft Building Footprints API not yet integrated")
    return _generate_synthetic_buildings(bbox, default_height)


def _generate_synthetic_buildings(
    bbox: Bbox,
    default_height: float,
    density: float = 0.3,
) -> BuildingDataset:
    """Generate synthetic building footprints for testing.

    Creates rectangular buildings with random placement,
    size, and height distribution typical of urban areas.
    """
    west, south, east, north = bbox
    rng = np.random.default_rng(42)

    # Approximate area in meters
    lat_mid = (south + north) / 2
    dx = (east - west) * 111320 * np.cos(np.radians(lat_mid))
    dy = (north - south) * 110540

    # Number of buildings based on density
    area_km2 = (dx * dy) / 1e6
    n_buildings = int(area_km2 * density * 500)  # ~150 buildings/km² for suburban
    n_buildings = min(n_buildings, 2000)  # cap for performance

    buildings = []
    for _ in range(n_buildings):
        # Random center
        cx = west + rng.uniform(0.1, 0.9) * (east - west)
        cy = south + rng.uniform(0.1, 0.9) * (north - south)

        # Random size (10-50m footprint)
        w = rng.uniform(8, 40) / (111320 * np.cos(np.radians(lat_mid)))
        h = rng.uniform(8, 30) / 110540

        # Rectangular footprint
        footprint = np.array([
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
            [cx - w / 2, cy - h / 2],  # close polygon
        ])

        # Height: log-normal distribution (most buildings short, few tall)
        height = rng.lognormal(mean=np.log(default_height), sigma=0.5)
        height = min(height, 100)  # cap at 100m

        btype = rng.choice(["residential", "commercial", "industrial"], p=[0.6, 0.3, 0.1])

        buildings.append(Building(
            footprint=footprint, height=height, building_type=btype,
        ))

    logger.info("Synthetic: generated %d buildings for %.1f km²", len(buildings), area_km2)
    return BuildingDataset(buildings=buildings, bbox=bbox, source="synthetic")


def extrude_buildings_to_mesh(
    dataset: BuildingDataset,
    dem: np.ndarray | None = None,
    pixel_size: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude building footprints into 3D box meshes.

    Each building becomes a box with the footprint as base and
    height extruded upward. Can be loaded as Genesis rigid bodies.

    Args:
        dataset: Building footprints.
        dem: Optional DEM for ground elevation lookup.
        pixel_size: DEM pixel size for coordinate mapping.

    Returns:
        Tuple of (vertices (N,3), faces (F,3)) for combined mesh.
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    for building in dataset.buildings:
        fp = building.footprint
        if len(fp) < 4:
            continue

        # Use first 4 points as rectangle corners
        corners = fp[:4]
        z_base = building.ground_elevation
        z_top = z_base + building.height

        # 8 vertices (4 bottom + 4 top)
        bottom = np.column_stack([corners, np.full(4, z_base)])
        top = np.column_stack([corners, np.full(4, z_top)])
        verts = np.vstack([bottom, top])

        # 12 faces (6 sides × 2 triangles each)
        faces = np.array([
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top
            [4, 6, 5], [4, 7, 6],
            # Front
            [0, 4, 5], [0, 5, 1],
            # Back
            [2, 6, 7], [2, 7, 3],
            # Left
            [0, 3, 7], [0, 7, 4],
            # Right
            [1, 5, 6], [1, 6, 2],
        ], dtype=np.int32) + vert_offset

        all_verts.append(verts)
        all_faces.append(faces)
        vert_offset += 8

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_faces).astype(np.int32),
    )
