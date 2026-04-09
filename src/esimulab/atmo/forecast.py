"""Weather forecast models via Earth2Studio (GraphCast, Pangu, FourCastNet)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)


def _try_import_model(model_name: str) -> Any:
    """Import an Earth2Studio prognostic model."""
    models = {
        "graphcast": "earth2studio.models.px.GraphCastOperational",
        "pangu": "earth2studio.models.px.Pangu6",
        "fcnv2": "earth2studio.models.px.FCNv2Small",
    }

    module_path = models.get(model_name)
    if module_path is None:
        logger.warning("Unknown model: %s", model_name)
        return None

    try:
        parts = module_path.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(parts[0])
        return getattr(mod, parts[1])
    except (ImportError, AttributeError):
        logger.warning("Model %s not available", model_name)
        return None


def run_forecast(
    time: datetime,
    model_name: str = "graphcast",
    nsteps: int = 4,
    variables: list[str] | None = None,
    output_path: str | None = None,
) -> xr.Dataset | None:
    """Run a weather forecast using an Earth2Studio prognostic model.

    Args:
        time: Initial time for the forecast.
        model_name: 'graphcast', 'pangu', or 'fcnv2'.
        nsteps: Number of forecast steps (6h each for GraphCast).
        variables: Output variables. Default: [t2m, u10m, v10m, tp].
        output_path: Optional zarr output path.

    Returns:
        xr.Dataset with forecast fields, or None if model unavailable.
    """
    variables = variables or ["t2m", "u10m", "v10m", "tp"]

    model_cls = _try_import_model(model_name)
    if model_cls is None:
        logger.warning("Forecast model %s not available, returning None", model_name)
        return None

    try:
        from earth2studio.data import GFS
        from earth2studio.io import ZarrBackend
        from earth2studio.run import deterministic

        logger.info("Running %s forecast: %d steps from %s", model_name, nsteps, time)

        package = model_cls.load_default_package()
        model = model_cls.load_model(package)

        io_backend = ZarrBackend(output_path) if output_path else ZarrBackend()

        output = deterministic(
            time=[time.isoformat()],
            nsteps=nsteps,
            prognostic=model,
            data=GFS(),
            io=io_backend,
            output_coords={"variable": np.array(variables)},
        )

        # Convert to Dataset
        ds = xr.Dataset()
        for var in variables:
            if var in output:
                ds[var] = xr.DataArray(output[var])

        # Free model VRAM
        import torch

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Forecast complete: %d variables, %d timesteps", len(ds.data_vars), nsteps)
        return ds

    except Exception:
        logger.exception("Forecast failed")
        return None


def generate_synthetic_forecast(
    time: datetime,
    nsteps: int = 4,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Generate synthetic forecast data for testing.

    Creates simple linear extrapolation as a placeholder.
    """
    variables = variables or ["t2m", "u10m", "v10m", "tp"]
    rng = np.random.default_rng(42)

    lat = np.arange(-90, 90, 2.5)
    lon = np.arange(0, 360, 2.5)
    steps = np.arange(nsteps)

    ds = xr.Dataset(coords={"lat": lat, "lon": lon, "step": steps})

    for var in variables:
        base = rng.normal(0, 1, (len(lat), len(lon))).astype(np.float32)
        # Linear drift over time
        data = np.stack(
            [base + i * rng.normal(0, 0.1, base.shape).astype(np.float32) for i in range(nsteps)]
        )
        ds[var] = xr.DataArray(data, dims=["step", "lat", "lon"])

    return ds
