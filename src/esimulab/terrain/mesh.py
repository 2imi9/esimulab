"""DEM-to-mesh conversion and decimation for Genesis.

Provides an alternative to direct heightfield loading: converts DEM
to a triangle mesh, optionally decimates for GPU efficiency, and
exports as .obj for gs.morphs.Mesh loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def dem_to_mesh(
    dem: np.ndarray,
    pixel_size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a DEM heightfield to a triangle mesh.

    Args:
        dem: (H, W) float32 heightfield array, NaN-free.
        pixel_size: Grid spacing in meters.
        origin: (x, y) offset for mesh origin.

    Returns:
        Tuple of (vertices (N, 3), faces (F, 3)) as numpy arrays.
    """
    nrows, ncols = dem.shape
    hf = np.nan_to_num(dem, nan=0.0).astype(np.float32)

    # Build vertex grid
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xs = origin[0] + cols.astype(np.float32) * pixel_size
    ys = origin[1] + rows.astype(np.float32) * pixel_size
    zs = hf

    vertices = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    # Build triangle faces (2 triangles per grid cell)
    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            idx = r * ncols + c
            # Triangle 1: top-left
            faces.append([idx, idx + ncols, idx + 1])
            # Triangle 2: bottom-right
            faces.append([idx + 1, idx + ncols, idx + ncols + 1])

    faces = np.array(faces, dtype=np.int32)

    logger.info(
        "DEM to mesh: %d vertices, %d faces from %dx%d grid",
        vertices.shape[0], faces.shape[0], nrows, ncols,
    )
    return vertices, faces


def dem_to_mesh_vectorized(
    dem: np.ndarray,
    pixel_size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized DEM-to-mesh conversion (faster for large grids).

    Same interface as dem_to_mesh but uses numpy broadcasting
    instead of Python loops.
    """
    nrows, ncols = dem.shape
    hf = np.nan_to_num(dem, nan=0.0).astype(np.float32)

    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xs = origin[0] + cols.astype(np.float32) * pixel_size
    ys = origin[1] + rows.astype(np.float32) * pixel_size

    vertices = np.column_stack([xs.ravel(), ys.ravel(), hf.ravel()])

    # Vectorized face generation
    r = np.arange(nrows - 1)[:, None]
    c = np.arange(ncols - 1)[None, :]
    idx = r * ncols + c

    tri1 = np.stack([idx, idx + ncols, idx + 1], axis=-1).reshape(-1, 3)
    tri2 = np.stack([idx + 1, idx + ncols, idx + ncols + 1], axis=-1).reshape(-1, 3)
    faces = np.vstack([tri1, tri2]).astype(np.int32)

    return vertices, faces


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_reduction: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a mesh using fast_simplification.

    Args:
        vertices: (N, 3) vertex array.
        faces: (F, 3) face index array.
        target_reduction: Fraction of faces to remove (0.7 = keep 30%).

    Returns:
        Tuple of (simplified_vertices, simplified_faces).
    """
    try:
        import fast_simplification

        verts_out, faces_out = fast_simplification.simplify_mesh(
            vertices, faces, target_reduction=target_reduction
        )
        logger.info(
            "Decimated mesh: %d→%d vertices, %d→%d faces (%.0f%% reduction)",
            vertices.shape[0], verts_out.shape[0],
            faces.shape[0], faces_out.shape[0],
            target_reduction * 100,
        )
        return verts_out, faces_out
    except ImportError:
        logger.warning("fast_simplification not installed, returning original mesh")
        return vertices, faces


def export_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str | Path,
) -> Path:
    """Export mesh as Wavefront OBJ file.

    Args:
        vertices: (N, 3) vertex positions.
        faces: (F, 3) face indices (0-based).
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# Esimulab terrain mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces\n")
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # OBJ is 1-indexed

    logger.info("Exported OBJ: %s (%d KB)", path, path.stat().st_size // 1024)
    return path


def export_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: str | Path,
    file_format: str | None = None,
) -> Path:
    """Export mesh using trimesh (supports .obj, .stl, .glb, .ply).

    Args:
        vertices: (N, 3) vertex positions.
        faces: (F, 3) face indices.
        path: Output file path (format inferred from extension).
        file_format: Override format ('obj', 'stl', 'glb', 'ply').

    Returns:
        Path to the written file.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type=file_format)

    logger.info("Exported %s: %s (%d KB)", path.suffix, path, path.stat().st_size // 1024)
    return path
