"""End-to-end pipeline test in Docker.

Globe selection → real DEM → real ERA5 → Genesis sim → particle export → web data.

Usage:
    docker compose --profile gpu run --rm genesis-sim uv run python scripts/run_e2e_pipeline.py
"""

from __future__ import annotations

import json
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    output_dir = Path("data")
    bbox = (-118.3, 34.0, -118.2, 34.1)  # LA foothills — real terrain with elevation
    sim_time = datetime(2023, 6, 15, 12, 0)
    num_steps = 50

    print("=" * 70)
    print("  ESIMULAB END-TO-END PIPELINE")
    print(f"  Region: {bbox}")
    print(f"  Time: {sim_time}")
    print(f"  Steps: {num_steps}")
    print("=" * 70)

    # ── Phase 1: Terrain ────────────────────────────────────
    print("\n[Phase 1] Fetching terrain...")
    t0 = time.time()

    from esimulab.terrain import fetch_dem, prepare_heightfield

    dem_result = fetch_dem(bbox)
    heightfield = prepare_heightfield(dem_result.heightfield, dem_result.pixel_size)

    terrain_dir = output_dir / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)
    np.save(terrain_dir / "heightfield.npy", heightfield.height_field)
    meta = {
        "pixel_size": heightfield.horizontal_scale,
        "vertical_scale": heightfield.vertical_scale,
        "rows": heightfield.height_field.shape[0],
        "cols": heightfield.height_field.shape[1],
        "origin": list(heightfield.origin),
        "bounds_min": list(heightfield.bounds_min),
        "bounds_max": list(heightfield.bounds_max),
        "bbox": list(bbox),
    }
    (terrain_dir / "metadata.json").write_text(json.dumps(meta))

    print(f"  [OK] DEM: {heightfield.height_field.shape}, "
          f"elev {heightfield.bounds_min[2]:.0f}-{heightfield.bounds_max[2]:.0f}m, "
          f"pixel {heightfield.horizontal_scale:.1f}m  ({time.time()-t0:.1f}s)")

    # ── Phase 2: Atmosphere ─────────────────────────────────
    print("\n[Phase 2] Fetching ERA5 atmospheric data...")
    t0 = time.time()

    from esimulab.atmo import extract_precip_rate, extract_wind_forcing, fetch_era5

    atmo_ds = fetch_era5(bbox, sim_time)
    wind = extract_wind_forcing(atmo_ds)
    precip = extract_precip_rate(atmo_ds)

    # Material properties from temperature
    from esimulab.atmo.material_mapping import materials_from_atmosphere

    t2m = float(atmo_ds["t2m"].mean().values)
    materials = materials_from_atmosphere(t2m)

    atmo_dir = output_dir / "atmo"
    atmo_dir.mkdir(parents=True, exist_ok=True)
    (atmo_dir / "wind.json").write_text(json.dumps({
        "direction": list(wind.direction),
        "magnitude": wind.magnitude,
        "turbulence_strength": wind.turbulence_strength,
        "source": "ERA5_ARCO",
    }))
    (atmo_dir / "precip.json").write_text(json.dumps({
        "rate_mm_hr": precip.rate_mm_hr,
        "terminal_velocity": precip.terminal_velocity,
        "droplet_size": precip.droplet_size,
    }))

    d = wind.direction
    print(f"  [OK] Wind: {wind.magnitude:.2f} m/s, dir=({d[0]:.2f}, {d[1]:.2f})")
    print(f"  [OK] Precip: {precip.rate_mm_hr:.2f} mm/hr")
    print(f"  [OK] Temp: {t2m:.1f} K ({t2m-273.15:.1f}°C)")
    print(f"  [OK] Materials: {materials.description}  ({time.time()-t0:.1f}s)")

    # ── Phase 3: Genesis Simulation ─────────────────────────
    print("\n[Phase 3] Building Genesis scene...")
    t0 = time.time()

    import genesis as gs

    gs.init(backend=gs.gpu, precision="32")

    # Downsample terrain for Genesis (large heightfields are slow)
    hf = heightfield.height_field
    max_gs_dim = 128
    if hf.shape[0] > max_gs_dim or hf.shape[1] > max_gs_dim:
        step_r = max(1, hf.shape[0] // max_gs_dim)
        step_c = max(1, hf.shape[1] // max_gs_dim)
        hf = hf[::step_r, ::step_c]
        pixel_size = heightfield.horizontal_scale * max(step_r, step_c)
        print(f"  Downsampled terrain: {hf.shape} (pixel {pixel_size:.1f}m)")
    else:
        pixel_size = heightfield.horizontal_scale

    bmin = heightfield.bounds_min
    bmax = heightfield.bounds_max
    margin = 50.0
    solver_lower = (bmin[0] - margin, bmin[1] - margin, bmin[2] - 10)
    solver_upper = (bmax[0] + margin, bmax[1] + margin, bmax[2] + 200)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        sph_options=gs.options.SPHOptions(
            lower_bound=solver_lower,
            upper_bound=solver_upper,
            particle_size=0.05,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(morph=gs.morphs.Terrain(
        height_field=hf,
        horizontal_scale=pixel_size,
        vertical_scale=heightfield.vertical_scale,
        pos=heightfield.origin,
    ))

    # Water body at mid-elevation
    z_mid = (bmin[2] + bmax[2]) / 2
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            rho=materials.water.rho,
            mu=materials.water.mu,
            gamma=materials.water.gamma,
        ),
        morph=gs.morphs.Box(
            pos=(0, 0, z_mid + 20),
            size=(50, 50, 10),
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.6, 1.0, 0.8), vis_mode="particle"),
    )

    # Wind force fields from ERA5
    if wind.magnitude > 0.1:
        scene.add_force_field(
            gs.engine.force_fields.Constant(
                direction=wind.direction, strength=wind.magnitude
            )
        )
        scene.add_force_field(
            gs.engine.force_fields.Turbulence(
                strength=wind.turbulence_strength, frequency=5
            )
        )

    scene.build()
    print(f"  [OK] Scene built  ({time.time()-t0:.1f}s)")

    # ── Phase 4: Run Simulation ─────────────────────────────
    print(f"\n[Phase 4] Running {num_steps} simulation steps...")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    export_interval = 5
    exported_frames = 0

    for step in range(num_steps):
        scene.step()

        if step % export_interval == 0:
            try:
                pos = water.get_pos().cpu().numpy()
                frame_path = frames_dir / f"frame_{step:06d}.bin"
                n = pos.shape[0]
                with open(frame_path, "wb") as f:
                    f.write(struct.pack("<I", n))
                    f.write(pos.astype(np.float32).tobytes())
                exported_frames += 1
            except Exception as e:
                if step == 0:
                    print(f"  [WARN] Particle export failed: {e}")

        if step % 10 == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step}/{num_steps} ({rate:.1f} steps/s)")

    elapsed = time.time() - t0
    print(f"  [OK] {num_steps} steps in {elapsed:.1f}s ({num_steps/elapsed:.1f} steps/s)")
    print(f"  [OK] Exported {exported_frames} particle frames")

    # ── Phase 5: Save metadata ──────────────────────────────
    (output_dir / "metadata.json").write_text(json.dumps({
        "bbox": list(bbox),
        "time": sim_time.isoformat(),
        "steps": num_steps,
        "wind_magnitude": wind.magnitude,
        "temperature_k": t2m,
        "precipitation_mm_hr": precip.rate_mm_hr,
        "source": "ERA5_ARCO",
        "materials": materials.description,
    }))

    # ── Verify ──────────────────────────────────────────────
    print("\n[Verify] Checking output files...")
    checks = [
        (terrain_dir / "heightfield.npy", "terrain heightfield"),
        (terrain_dir / "metadata.json", "terrain metadata"),
        (atmo_dir / "wind.json", "wind data"),
        (atmo_dir / "precip.json", "precip data"),
        (output_dir / "metadata.json", "pipeline metadata"),
    ]

    all_ok = True
    for path, label in checks:
        if path.exists():
            print(f"  [OK] {label}: {path.stat().st_size:,} bytes")
        else:
            print(f"  [FAIL] {label}: missing")
            all_ok = False

    frame_count = len(list(frames_dir.glob("frame_*.bin")))
    print(f"  [OK] particle frames: {frame_count} files")

    print("\n" + "=" * 70)
    if all_ok and frame_count > 0:
        print("  E2E PIPELINE: ALL PASSED")
        print("  Run 'esimulab serve' to view results in browser")
    else:
        print("  E2E PIPELINE: SOME CHECKS FAILED")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
