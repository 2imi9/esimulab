"""WebSocket streaming for live simulation frame updates."""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import WebSocket

logger = logging.getLogger(__name__)


class FrameStreamer:
    """Streams particle frames to connected WebSocket clients."""

    def __init__(self, max_particles_per_frame: int = 50000):
        self.clients: list[WebSocket] = []
        self.max_particles = max_particles_per_frame

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(self.clients))

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(self.clients))

    async def broadcast_frame(self, positions: np.ndarray) -> None:
        """Send particle positions to all connected clients.

        Args:
            positions: (N, 3) float32 array of particle positions.
        """
        if not self.clients:
            return

        # Subsample if too many particles
        positions = subsample_particles(positions, self.max_particles)

        # Pack as binary: [num_particles (uint32)] [x,y,z,...] (float32)
        n = positions.shape[0]
        data = struct.pack("<I", n) + positions.astype(np.float32).tobytes()

        disconnected = []
        for client in self.clients:
            try:
                await client.send_bytes(data)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self.disconnect(client)

    async def broadcast_metadata(self, metadata: dict) -> None:
        """Send JSON metadata to all connected clients."""
        import json

        msg = json.dumps(metadata)
        disconnected = []
        for client in self.clients:
            try:
                await client.send_text(msg)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self.disconnect(client)


def subsample_particles(
    positions: np.ndarray,
    max_count: int = 50000,
) -> np.ndarray:
    """Subsample particle positions for web viewer performance.

    Uses uniform random sampling to reduce particle count while
    maintaining spatial distribution.

    Args:
        positions: (N, 3) float32 array.
        max_count: Maximum particles to return.

    Returns:
        (M, 3) array where M <= max_count.
    """
    n = positions.shape[0]
    if n <= max_count:
        return positions

    rng = np.random.default_rng(0)  # deterministic for consistent visuals
    indices = rng.choice(n, size=max_count, replace=False)
    indices.sort()  # maintain spatial coherence
    return positions[indices]


def load_and_subsample_frame(
    frame_path: Path,
    max_particles: int = 50000,
) -> np.ndarray | None:
    """Load a binary frame file and subsample for web delivery.

    Args:
        frame_path: Path to .bin frame file.
        max_particles: Maximum particles to return.

    Returns:
        (M, 3) float32 array, or None if file invalid.
    """
    try:
        with open(frame_path, "rb") as f:
            (n,) = struct.unpack("<I", f.read(4))
            data = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
        return subsample_particles(data, max_particles)
    except Exception:
        logger.debug("Failed to load frame %s", frame_path, exc_info=True)
        return None
