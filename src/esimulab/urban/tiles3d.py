"""Google 3D Tiles integration for photogrammetric city meshes.

Google's Map Tiles API provides photogrammetry-derived 3D meshes
of major cities as OGC 3D Tiles (glTF/GLB). These meshes include
buildings, bridges, and infrastructure at ~1m resolution.

Data flow:
  1. Query Map Tiles API for tileset.json at bbox
  2. Traverse tile tree to find tiles intersecting region
  3. Download GLB meshes for matching tiles
  4. Convert to Genesis-compatible mesh (trimesh)
  5. Load as rigid bodies in Genesis scene

API: https://developers.google.com/maps/documentation/tile/3d-tiles
Free tier: 100k tile requests/month
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]  # (west, south, east, north)

# Google Map Tiles API endpoint for 3D tiles
TILES_API_ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json"


@dataclass
class Tile3DConfig:
    """Configuration for 3D tile fetching."""

    api_key: str = ""  # Google Maps API key
    max_tiles: int = 50  # max tiles to download
    min_geometric_error: float = 10.0  # minimum detail level (meters)
    cache_dir: str = "data/tiles3d_cache"
    formats: list[str] = field(default_factory=lambda: ["glb", "gltf"])


@dataclass
class CityMesh:
    """A city mesh from 3D tiles."""

    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (F, 3) int32
    bounds: Bbox  # geographic bounds
    source: str = "google_3d_tiles"
    tile_count: int = 0
    total_triangles: int = 0


def fetch_3d_tiles(
    bbox: Bbox,
    config: Tile3DConfig | None = None,
) -> CityMesh | None:
    """Fetch Google 3D Tiles for a bounding box.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        config: Tile fetching configuration.

    Returns:
        CityMesh with combined vertices/faces, or None if unavailable.
    """
    config = config or Tile3DConfig()

    if not config.api_key:
        logger.warning(
            "No Google Maps API key provided. Set GOOGLE_MAPS_API_KEY env var "
            "or pass api_key in Tile3DConfig. Falling back to synthetic city."
        )
        return generate_synthetic_city_mesh(bbox)

    try:
        return _fetch_google_3d_tiles(bbox, config)
    except Exception:
        logger.exception("Google 3D Tiles fetch failed, using synthetic fallback")
        return generate_synthetic_city_mesh(bbox)


def _fetch_google_3d_tiles(
    bbox: Bbox,
    config: Tile3DConfig,
) -> CityMesh | None:
    """Fetch tiles from Google Map Tiles API."""
    import requests

    west, south, east, north = bbox

    # Step 1: Get root tileset
    root_url = f"{TILES_API_ROOT}?key={config.api_key}"
    resp = requests.get(root_url, timeout=30)
    resp.raise_for_status()
    tileset = resp.json()

    # Step 2: Traverse tile tree
    tiles_to_fetch = _find_tiles_in_bbox(
        tileset, bbox, config.min_geometric_error, config.max_tiles
    )
    logger.info("Found %d tiles intersecting bbox", len(tiles_to_fetch))

    if not tiles_to_fetch:
        return None

    # Step 3: Download and parse GLB meshes
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_verts = []
    all_faces = []
    vert_offset = 0

    for tile_url in tiles_to_fetch:
        try:
            mesh_data = _download_tile_mesh(tile_url, config.api_key, cache_dir)
            if mesh_data is not None:
                verts, faces = mesh_data
                faces = faces + vert_offset
                all_verts.append(verts)
                all_faces.append(faces)
                vert_offset += verts.shape[0]
        except Exception:
            logger.debug("Tile download failed: %s", tile_url, exc_info=True)

    if not all_verts:
        return None

    combined_verts = np.vstack(all_verts).astype(np.float32)
    combined_faces = np.vstack(all_faces).astype(np.int32)

    return CityMesh(
        vertices=combined_verts,
        faces=combined_faces,
        bounds=bbox,
        tile_count=len(all_verts),
        total_triangles=combined_faces.shape[0],
    )


def _find_tiles_in_bbox(
    tileset: dict,
    bbox: Bbox,
    min_error: float,
    max_tiles: int,
) -> list[str]:
    """Traverse 3D Tiles tree to find tiles intersecting bbox."""
    urls = []

    def _traverse(node: dict, depth: int = 0) -> None:
        if len(urls) >= max_tiles:
            return

        # Check geometric error
        error = node.get("geometricError", float("inf"))
        if error < min_error:
            return

        # Check bounding volume intersection
        bv = node.get("boundingVolume", {})
        if not _bv_intersects_bbox(bv, bbox):
            return

        # Collect content URL
        content = node.get("content", {})
        uri = content.get("uri", content.get("url", ""))
        if uri and any(uri.endswith(f".{fmt}") for fmt in ["glb", "gltf", "b3dm"]):
            urls.append(uri)

        # Recurse children
        for child in node.get("children", []):
            _traverse(child, depth + 1)

    root = tileset.get("root", tileset)
    _traverse(root)
    return urls


def _bv_intersects_bbox(bv: dict, bbox: Bbox) -> bool:
    """Check if a 3D Tiles bounding volume intersects a geographic bbox."""
    # Region bounding volume: [west, south, east, north, minHeight, maxHeight] in radians
    region = bv.get("region")
    if region and len(region) >= 4:
        bv_west = np.degrees(region[0])
        bv_south = np.degrees(region[1])
        bv_east = np.degrees(region[2])
        bv_north = np.degrees(region[3])
        west, south, east, north = bbox
        return not (bv_east < west or bv_west > east or bv_north < south or bv_south > north)

    # Box or sphere — assume intersection (conservative)
    return True


def _download_tile_mesh(
    url: str,
    api_key: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Download a single tile mesh and parse as trimesh."""
    import hashlib

    import requests

    # Cache by URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_path = cache_dir / f"{url_hash}.glb"

    if cache_path.exists():
        data = cache_path.read_bytes()
    else:
        full_url = f"{url}?key={api_key}" if "key=" not in url else url
        resp = requests.get(full_url, timeout=30)
        resp.raise_for_status()
        data = resp.content
        cache_path.write_bytes(data)

    # Parse with trimesh
    import trimesh

    mesh = trimesh.load(
        trimesh.util.wrap_as_stream(data),
        file_type="glb",
        process=False,
    )

    if isinstance(mesh, trimesh.Scene):
        # Combine all geometries in the scene
        combined = trimesh.util.concatenate(list(mesh.geometry.values()))
        return combined.vertices.astype(np.float32), combined.faces.astype(np.int32)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)

    return None


def generate_synthetic_city_mesh(
    bbox: Bbox,
    building_count: int = 200,
    seed: int = 42,
) -> CityMesh:
    """Generate synthetic city mesh for testing without API key.

    Creates a grid of box buildings with varied heights,
    simulating a basic urban layout.
    """
    rng = np.random.default_rng(seed)
    west, south, east, north = bbox

    lat_mid = (south + north) / 2
    dx = (east - west) * 111320 * np.cos(np.radians(lat_mid))
    dy = (north - south) * 110540

    all_verts = []
    all_faces = []
    offset = 0

    for _ in range(building_count):
        # Random position within bbox (in meters from center)
        bx = rng.uniform(-dx / 2, dx / 2)
        by = rng.uniform(-dy / 2, dy / 2)
        bw = rng.uniform(10, 40)  # width
        bd = rng.uniform(10, 30)  # depth
        bh = rng.lognormal(mean=np.log(15), sigma=0.6)  # height
        bh = min(bh, 80)

        # 8 vertices for a box
        verts = np.array(
            [
                [bx - bw / 2, by - bd / 2, 0],
                [bx + bw / 2, by - bd / 2, 0],
                [bx + bw / 2, by + bd / 2, 0],
                [bx - bw / 2, by + bd / 2, 0],
                [bx - bw / 2, by - bd / 2, bh],
                [bx + bw / 2, by - bd / 2, bh],
                [bx + bw / 2, by + bd / 2, bh],
                [bx - bw / 2, by + bd / 2, bh],
            ],
            dtype=np.float32,
        )

        faces = (
            np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],  # bottom
                    [4, 6, 5],
                    [4, 7, 6],  # top
                    [0, 4, 5],
                    [0, 5, 1],  # front
                    [2, 6, 7],
                    [2, 7, 3],  # back
                    [0, 3, 7],
                    [0, 7, 4],  # left
                    [1, 5, 6],
                    [1, 6, 2],  # right
                ],
                dtype=np.int32,
            )
            + offset
        )

        all_verts.append(verts)
        all_faces.append(faces)
        offset += 8

    combined_verts = np.vstack(all_verts)
    combined_faces = np.vstack(all_faces)

    logger.info(
        "Synthetic city: %d buildings, %d verts, %d faces",
        building_count,
        combined_verts.shape[0],
        combined_faces.shape[0],
    )

    return CityMesh(
        vertices=combined_verts,
        faces=combined_faces,
        bounds=bbox,
        source="synthetic",
        tile_count=building_count,
        total_triangles=combined_faces.shape[0],
    )


def city_mesh_to_genesis(
    gs: Any,
    scene: Any,
    mesh: CityMesh,
    export_path: str | Path | None = None,
) -> Any | None:
    """Load a city mesh into Genesis as a rigid body.

    Args:
        gs: Genesis module.
        scene: Genesis scene.
        mesh: City mesh data.
        export_path: Optional path to save .obj for Genesis.

    Returns:
        Genesis entity, or None if failed.
    """
    if mesh.vertices.shape[0] == 0:
        return None

    # Export as OBJ for Genesis
    if export_path is None:
        export_path = Path("data/city_mesh.obj")

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    from esimulab.terrain.mesh import export_obj

    export_obj(mesh.vertices, mesh.faces, export_path)

    # Load into Genesis
    try:
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(export_path),
                fixed=True,
                scale=1.0,
            ),
        )
        logger.info(
            "City mesh loaded into Genesis: %d triangles from %s",
            mesh.total_triangles,
            mesh.source,
        )
        return entity
    except Exception:
        logger.warning("Failed to load city mesh into Genesis", exc_info=True)
        return None
