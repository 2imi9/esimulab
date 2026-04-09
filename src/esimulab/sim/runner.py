"""Simulation execution and frame export."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from esimulab.atmo.precip import PrecipForcing

logger = logging.getLogger(__name__)


def _export_frame(positions: np.ndarray, path: Path) -> None:
    """Write particle positions as a binary Float32Array file.

    Format: [num_particles (uint32)] [x,y,z,x,y,z,...] (float32)
    """
    n = positions.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(positions.astype(np.float32).tobytes())


def run_simulation(
    components: dict[str, Any],
    precip: PrecipForcing | None = None,
    num_steps: int = 600,
    export_dir: str | Path | None = None,
    export_interval: int = 10,
    video_path: str | None = None,
    video_fps: int = 30,
) -> None:
    """Run the Genesis simulation loop with frame export.

    Args:
        components: Dict from build_scene() with 'scene', 'emitter', 'camera'.
        precip: Precipitation parameters for rain emission.
        num_steps: Total simulation steps.
        export_dir: Directory for binary particle frame export. None to skip.
        export_interval: Export a frame every N steps.
        video_path: Path for MP4 video output. None to skip recording.
        video_fps: Video frames per second.
    """
    scene = components["scene"]
    emitter = components["emitter"]
    camera = components["camera"]

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    if video_path:
        camera.start_recording()

    logger.info("Starting simulation: %d steps", num_steps)

    for step in range(num_steps):
        # Emit rain particles if active
        if emitter is not None and precip is not None and precip.rate_mm_hr > 0 and step % 5 == 0:
            emitter.emit(
                droplet_shape="circle",
                droplet_size=precip.droplet_size,
                pos=(0.0, 0.0, float(components.get("z_top", 100.0))),
                direction=(0, 0, -1),
                speed=precip.terminal_velocity,
                theta=0.3,  # slight spread cone
            )

        scene.step()
        camera.render()

        # Export particle positions
        if export_dir and step % export_interval == 0:
            try:
                # Get SPH particle positions via entity API
                for entity in components.get("sph_entities", []):
                    if hasattr(entity, "get_particles_pos"):
                        pos = entity.get_particles_pos().cpu().numpy()
                        frame_path = export_dir / f"frame_{step:06d}.bin"
                        _export_frame(pos, frame_path)
                        break
            except Exception:
                logger.debug("Could not export frame %d", step, exc_info=True)

        if step % 100 == 0:
            logger.info("Step %d / %d", step, num_steps)

    if video_path:
        camera.stop_recording(save_to_filename=video_path, fps=video_fps)
        logger.info("Video saved to %s", video_path)

    logger.info("Simulation complete: %d steps", num_steps)
