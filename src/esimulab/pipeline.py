"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(
    bbox: tuple[float, float, float, float],
    time: datetime,
    num_steps: int = 600,
    output_dir: Path | None = None,
    skip_gpu: bool = False,
    backend: str | None = None,
    serve: bool = False,
    port: int = 8000,
) -> None:
    """Run the full Esimulab pipeline: terrain -> atmo -> sim -> web.

    Args:
        bbox: (west, south, east, north) in EPSG:4326.
        time: Target datetime for atmospheric data.
        num_steps: Simulation steps.
        output_dir: Output directory for results.
        skip_gpu: If True, skip Genesis simulation (data-only mode).
        backend: Genesis compute backend ('gpu', 'cpu', or None for auto).
        serve: If True, launch web viewer after completion.
        port: Web viewer port.
    """
    from pathlib import Path

    output_dir = Path(output_dir or "data")

    # --- Phase 1: Terrain ---
    logger.info("Phase 1: Fetching terrain data")
    from esimulab.terrain import fetch_dem, prepare_heightfield

    dem_result = fetch_dem(bbox)
    heightfield = prepare_heightfield(dem_result.heightfield, dem_result.pixel_size)

    # Save terrain for web viewer
    terrain_dir = output_dir / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    np.save(terrain_dir / "heightfield.npy", heightfield.height_field)
    (terrain_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pixel_size": heightfield.horizontal_scale,
                "vertical_scale": heightfield.vertical_scale,
                "rows": heightfield.height_field.shape[0],
                "cols": heightfield.height_field.shape[1],
                "origin": list(heightfield.origin),
                "bounds_min": list(heightfield.bounds_min),
                "bounds_max": list(heightfield.bounds_max),
            }
        )
    )
    logger.info("Terrain saved to %s", terrain_dir)

    # --- Phase 2: Atmosphere ---
    logger.info("Phase 2: Fetching atmospheric data")
    from esimulab.atmo import extract_precip_rate, extract_wind_forcing, fetch_era5

    atmo_ds = fetch_era5(bbox, time)
    wind = extract_wind_forcing(atmo_ds)
    precip = extract_precip_rate(atmo_ds)

    # Save atmospheric metadata
    atmo_dir = output_dir / "atmo"
    atmo_dir.mkdir(parents=True, exist_ok=True)
    (atmo_dir / "wind.json").write_text(
        json.dumps(
            {
                "direction": list(wind.direction),
                "magnitude": wind.magnitude,
                "turbulence_strength": wind.turbulence_strength,
            }
        )
    )
    (atmo_dir / "precip.json").write_text(
        json.dumps(
            {
                "rate_mm_hr": precip.rate_mm_hr,
                "terminal_velocity": precip.terminal_velocity,
                "droplet_size": precip.droplet_size,
            }
        )
    )
    logger.info("Atmosphere saved to %s", atmo_dir)

    # --- Phase 3: Simulation (GPU required) ---
    if not skip_gpu:
        logger.info("Phase 3: Running Genesis simulation (%d steps)", num_steps)
        try:
            from esimulab.sim.runner import run_simulation
            from esimulab.sim.scene import build_scene

            components = build_scene(heightfield, wind=wind, precip=precip, backend=backend)
            run_simulation(
                components,
                precip=precip,
                num_steps=num_steps,
                export_dir=output_dir / "frames",
                video_path=str(output_dir / "video" / "simulation.mp4"),
            )
        except ImportError:
            logger.warning("Genesis not available — skipping simulation")
        except Exception:
            logger.exception("Simulation failed")
    else:
        logger.info("Phase 3: Skipped (--no-gpu)")

    # Save pipeline metadata
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "bbox": list(bbox),
                "time": time.isoformat(),
                "steps": num_steps,
                "skip_gpu": skip_gpu,
            }
        )
    )

    # --- Phase 4: Web viewer ---
    if serve:
        logger.info("Phase 4: Launching web viewer on port %d", port)
        import uvicorn

        # Point server at our output dir
        import esimulab.web.server as srv

        srv.DATA_DIR = output_dir

        uvicorn.run(srv.app, host="0.0.0.0", port=port)
    else:
        logger.info("Pipeline complete. Run 'esimulab --serve' to view results.")
