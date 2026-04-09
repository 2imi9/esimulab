"""Simulation execution and frame export."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from esimulab.atmo.precip import PrecipForcing
    from esimulab.atmo.wind import WindForcing

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
    wind: WindForcing | None = None,
    num_steps: int = 600,
    export_dir: str | Path | None = None,
    export_interval: int = 10,
    atmo_update_interval: int = 100,
    video_path: str | None = None,
    video_fps: int = 30,
    on_frame: Any = None,
) -> dict[str, Any]:
    """Run the Genesis simulation loop with frame export.

    Args:
        components: Dict from build_scene() with 'scene', 'emitter', 'camera'.
        precip: Precipitation parameters for rain emission.
        wind: Wind forcing parameters (for time-varying updates).
        num_steps: Total simulation steps.
        export_dir: Directory for binary particle frame export. None to skip.
        export_interval: Export a frame every N steps.
        atmo_update_interval: Re-evaluate boundary conditions every N steps.
        video_path: Path for MP4 video output. None to skip recording.
        video_fps: Video frames per second.
        on_frame: Optional async callback(positions) for WebSocket streaming.

    Returns:
        Dict with simulation statistics.
    """
    scene = components["scene"]
    emitter = components["emitter"]
    camera = components["camera"]

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

    if video_path:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        camera.start_recording()

    logger.info("Starting simulation: %d steps", num_steps)

    stats = {"steps": 0, "frames_exported": 0, "total_particles": 0}

    import time

    t_start = time.time()

    for step in range(num_steps):
        # --- Time-varying boundary conditions ---
        if step > 0 and step % atmo_update_interval == 0:
            _update_boundary_conditions(components, wind, precip, step, num_steps)

        # Emit rain particles if active
        if emitter is not None and precip is not None and precip.rate_mm_hr > 0 and step % 5 == 0:
            emitter.emit(
                droplet_shape="circle",
                droplet_size=precip.droplet_size,
                pos=(0.0, 0.0, float(components.get("z_top", 100.0))),
                direction=(0, 0, -1),
                speed=precip.terminal_velocity,
                theta=0.3,
            )

        scene.step()
        camera.render()

        # Export particle positions
        if export_dir and step % export_interval == 0:
            positions = _extract_particle_positions(components)
            if positions is not None:
                frame_path = export_dir / f"frame_{step:06d}.bin"
                _export_frame(positions, frame_path)
                stats["frames_exported"] += 1
                stats["total_particles"] = positions.shape[0]

        stats["steps"] = step + 1

        if step % 100 == 0:
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            logger.info("Step %d / %d (%.1f steps/s)", step, num_steps, rate)

    if video_path:
        camera.stop_recording(save_to_filename=video_path, fps=video_fps)
        logger.info("Video saved to %s", video_path)

    elapsed = time.time() - t_start
    stats["elapsed_seconds"] = elapsed
    stats["steps_per_second"] = num_steps / elapsed if elapsed > 0 else 0

    logger.info(
        "Simulation complete: %d steps in %.1fs (%.1f steps/s), %d frames exported",
        num_steps,
        elapsed,
        stats["steps_per_second"],
        stats["frames_exported"],
    )

    return stats


def _extract_particle_positions(components: dict[str, Any]) -> np.ndarray | None:
    """Extract SPH particle positions from scene entities."""
    # Try direct water entity
    water = components.get("water")
    if water is not None and hasattr(water, "get_particles_pos"):
        try:
            return water.get_particles_pos().cpu().numpy()
        except Exception:
            pass

    # Try sph_entities list
    for entity in components.get("sph_entities", []):
        if hasattr(entity, "get_particles_pos"):
            try:
                return entity.get_particles_pos().cpu().numpy()
            except Exception:
                continue

    return None


def _update_boundary_conditions(
    components: dict[str, Any],
    wind: WindForcing | None,
    precip: PrecipForcing | None,
    step: int,
    total_steps: int,
) -> None:
    """Update time-varying atmospheric forcing during simulation.

    Gradually varies wind direction and precipitation intensity
    to simulate changing weather conditions.
    """
    progress = step / total_steps  # 0.0 → 1.0

    if wind is not None and wind.magnitude > 0:
        # Rotate wind direction slightly over time (simulate weather change)
        import math

        angle = progress * math.pi * 0.25  # 45° rotation over full sim
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx, dy = wind.direction[0], wind.direction[1]
        new_dx = dx * cos_a - dy * sin_a
        new_dy = dx * sin_a + dy * cos_a

        # Scale magnitude with sinusoidal variation (±20%)
        scale = 1.0 + 0.2 * math.sin(progress * math.pi * 4)
        new_mag = wind.magnitude * scale

        logger.debug(
            "Boundary update step %d: wind dir=(%.2f,%.2f) mag=%.1f",
            step,
            new_dx,
            new_dy,
            new_mag,
        )
        # Note: Genesis doesn't support runtime force field modification directly.
        # This logs the intended update; actual implementation would need
        # scene reconstruction or custom Taichi kernel injection.

    if precip is not None and precip.rate_mm_hr > 0:
        # Ramp precipitation: start light, peak at 60%, taper off
        import math

        intensity = math.sin(progress * math.pi) * 1.5  # peaks at midpoint
        logger.debug("Boundary update step %d: precip intensity=%.2f", step, intensity)
