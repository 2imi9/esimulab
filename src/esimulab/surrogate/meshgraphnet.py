"""MeshGraphNet surrogate for terrain-aware simulation.

Uses PhysicsNeMo MeshGraphNet on unstructured terrain meshes:
  Input nodes:  (elevation, slope_x, slope_y, soil_type, velocity_x, velocity_y)
  Input edges:  (relative_dx, relative_dy, distance)
  Output:       (water_depth, sediment_flux, velocity_magnitude)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MGNConfig:
    """Configuration for MeshGraphNet surrogate."""

    input_dim_nodes: int = 6
    input_dim_edges: int = 3
    output_dim: int = 3
    processor_size: int = 15
    hidden_dim: int = 128
    num_layers_processor: int = 2
    aggregation: str = "sum"
    norm_type: str = "LayerNorm"


def _try_import_meshgraphnet():
    """Import PhysicsNeMo MeshGraphNet."""
    try:
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        return MeshGraphNet
    except ImportError:
        return None


def create_meshgraphnet(config: MGNConfig | None = None) -> Any:
    """Create a MeshGraphNet model for terrain-aware simulation.

    Args:
        config: Model configuration.

    Returns:
        MeshGraphNet model or None if PhysicsNeMo unavailable.
    """
    config = config or MGNConfig()

    mgn_cls = _try_import_meshgraphnet()
    if mgn_cls is None:
        logger.warning("PhysicsNeMo MeshGraphNet not available")
        return None

    model = mgn_cls(
        input_dim_nodes=config.input_dim_nodes,
        input_dim_edges=config.input_dim_edges,
        output_dim=config.output_dim,
        processor_size=config.processor_size,
        hidden_dim_processor=config.hidden_dim,
        hidden_dim_node_encoder=config.hidden_dim,
        hidden_dim_edge_encoder=config.hidden_dim,
        hidden_dim_node_decoder=config.hidden_dim,
        num_layers_node_processor=config.num_layers_processor,
        num_layers_edge_processor=config.num_layers_processor,
        aggregation=config.aggregation,
        norm_type=config.norm_type,
    )

    logger.info(
        "Created MeshGraphNet: %d→%d, %d message-passing layers",
        config.input_dim_nodes,
        config.output_dim,
        config.processor_size,
    )
    return model


def terrain_to_graph(
    dem: np.ndarray,
    pixel_size: float,
    max_nodes: int = 50000,
) -> dict[str, np.ndarray]:
    """Convert a DEM heightfield to graph representation for MeshGraphNet.

    Args:
        dem: (H, W) heightfield array.
        pixel_size: Meters per pixel.
        max_nodes: Maximum nodes (subsample if exceeded).

    Returns:
        Dict with 'node_features', 'edge_index', 'edge_features'.
    """
    h, w = dem.shape

    # Subsample if too large
    step = max(1, int(np.sqrt(h * w / max_nodes)))
    dem_sub = dem[::step, ::step]
    sh, sw = dem_sub.shape
    ps = pixel_size * step

    # Node positions and features
    rows, cols = np.mgrid[:sh, :sw]
    x = cols.ravel().astype(np.float32) * ps
    y = rows.ravel().astype(np.float32) * ps
    z = dem_sub.ravel().astype(np.float32)
    n_nodes = len(z)

    # Compute slopes
    dy, dx = np.gradient(dem_sub, ps)
    slope_x = dx.ravel().astype(np.float32)
    slope_y = dy.ravel().astype(np.float32)

    # Node features: [elevation, slope_x, slope_y, soil_type_proxy, vx, vy]
    node_features = np.stack(
        [
            z,
            slope_x,
            slope_y,
            np.zeros(n_nodes, dtype=np.float32),  # soil type placeholder
            np.zeros(n_nodes, dtype=np.float32),  # velocity x
            np.zeros(n_nodes, dtype=np.float32),  # velocity y
        ],
        axis=1,
    )

    # Build edges (4-connectivity grid)
    src, dst = [], []
    for r in range(sh):
        for c in range(sw):
            idx = r * sw + c
            if c + 1 < sw:
                neighbor = r * sw + (c + 1)
                src.extend([idx, neighbor])
                dst.extend([neighbor, idx])
            if r + 1 < sh:
                neighbor = (r + 1) * sw + c
                src.extend([idx, neighbor])
                dst.extend([neighbor, idx])

    edge_index = np.array([src, dst], dtype=np.int64)

    # Edge features: [dx, dy, distance]
    dx_e = x[edge_index[1]] - x[edge_index[0]]
    dy_e = y[edge_index[1]] - y[edge_index[0]]
    dist = np.sqrt(dx_e**2 + dy_e**2)
    edge_features = np.stack([dx_e, dy_e, dist], axis=1).astype(np.float32)

    logger.info(
        "Terrain graph: %d nodes, %d edges (step=%d from %dx%d)",
        n_nodes,
        edge_index.shape[1],
        step,
        h,
        w,
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }
