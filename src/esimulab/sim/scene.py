"""Genesis scene construction and configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from esimulab.atmo.precip import PrecipForcing
    from esimulab.atmo.wind import WindForcing
    from esimulab.terrain.convert import GenesisHeightfield

logger = logging.getLogger(__name__)


def _import_genesis():
    """Import genesis with clear error messaging."""
    try:
        import genesis as gs

        return gs
    except ImportError as e:
        msg = (
            "Genesis not installed. Install with: pip install genesis-world. "
            "GPU (CUDA) required for simulation."
        )
        raise ImportError(msg) from e


def build_scene(
    heightfield: GenesisHeightfield,
    wind: WindForcing | None = None,
    precip: PrecipForcing | None = None,
    dt: float = 2e-3,
    substeps: int = 10,
    sph_particle_size: float = 0.02,
    sf_resolution: int = 64,
    max_rain_particles: int = 100000,
    renderer_spp: int = 64,
    show_viewer: bool = False,
) -> dict[str, Any]:
    """Build a Genesis scene with terrain, water, and atmospheric forcing.

    Args:
        heightfield: Prepared Genesis heightfield from terrain pipeline.
        wind: Wind forcing parameters. If None, no wind applied.
        precip: Precipitation forcing. If None, no rain emitter.
        dt: Simulation timestep (seconds).
        substeps: Physics substeps per step.
        sph_particle_size: SPH particle radius.
        sf_resolution: Stable Fluid solver grid resolution.
        max_rain_particles: Maximum SPH particles for rain emitter.
        renderer_spp: Ray tracer samples per pixel.
        show_viewer: Whether to open the interactive viewer.

    Returns:
        Dict with keys: 'scene', 'terrain', 'emitter', 'camera'.
    """
    gs = _import_genesis()

    gs.init(backend=gs.gpu, precision="32")

    # Compute solver bounds from heightfield
    bmin = heightfield.bounds_min
    bmax = heightfield.bounds_max
    margin = 50.0  # meters beyond terrain bounds

    solver_lower = (bmin[0] - margin, bmin[1] - margin, bmin[2] - 10)
    solver_upper = (bmax[0] + margin, bmax[1] + margin, bmax[2] + 200)

    # Build scene options
    scene_kwargs = {
        "sim_options": gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0, 0, -9.81)),
        "rigid_options": gs.options.RigidOptions(enable_collision=True),
        "sph_options": gs.options.SPHOptions(
            lower_bound=solver_lower,
            upper_bound=solver_upper,
            particle_size=sph_particle_size,
        ),
        "coupler_options": gs.options.CouplerOptions(
            rigid_sph=True,
            rigid_mpm=True,
            mpm_sph=True,
        ),
        "renderer": gs.renderers.RayTracer(
            tracing_depth=32,
            lights=[
                {"pos": (0, 0, bmax[2] + 100), "color": (1, 1, 1), "intensity": 10, "radius": 4}
            ],
        ),
        "show_viewer": show_viewer,
    }

    # Only add SF solver if wind forcing provided
    if wind and wind.magnitude > 0:
        scene_kwargs["sf_options"] = gs.options.SFOptions(
            res=sf_resolution,
            inlet_vel=wind.direction,
            inlet_s=wind.magnitude * 20,  # scale for solver
        )

    scene = gs.Scene(**scene_kwargs)

    # Add terrain
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            height_field=heightfield.height_field,
            horizontal_scale=heightfield.horizontal_scale,
            vertical_scale=heightfield.vertical_scale,
            pos=heightfield.origin,
        ),
    )

    # Add wind force fields
    if wind and wind.magnitude > 0:
        scene.add_force_field(
            gs.engine.force_fields.Constant(direction=wind.direction, strength=wind.magnitude)
        )
        scene.add_force_field(
            gs.engine.force_fields.Turbulence(
                strength=wind.turbulence_strength,
                frequency=wind.turbulence_frequency,
            )
        )

    # Add rain emitter
    emitter = None
    if precip and precip.rate_mm_hr > 0:
        emitter = scene.add_emitter(
            material=gs.materials.SPH.Liquid(rho=1000.0, mu=0.002, gamma=0.005),
            max_particles=max_rain_particles,
            surface=gs.surfaces.Default(color=(0.5, 0.7, 1.0, 0.8)),
        )

    # Add camera
    center_x = (bmin[0] + bmax[0]) / 2
    center_y = (bmin[1] + bmax[1]) / 2
    center_z = (bmin[2] + bmax[2]) / 2
    extent = max(bmax[0] - bmin[0], bmax[1] - bmin[1])

    camera = scene.add_camera(
        res=(1920, 1080),
        pos=(center_x + extent * 0.8, center_y - extent * 0.6, bmax[2] + extent * 0.3),
        lookat=(center_x, center_y, center_z),
        fov=45,
        spp=renderer_spp,
    )

    scene.build()

    logger.info(
        "Scene built: terrain=%s, wind=%s, rain=%s",
        heightfield.height_field.shape,
        wind is not None,
        emitter is not None,
    )

    return {
        "scene": scene,
        "terrain": terrain,
        "emitter": emitter,
        "camera": camera,
    }
