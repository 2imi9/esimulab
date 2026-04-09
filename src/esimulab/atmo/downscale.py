"""AI downscaling of atmospheric data (CorrDiff, cBottle)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)


def _try_import_corrdiff():
    """Import CorrDiff model, returning None if unavailable."""
    try:
        from earth2studio.models.dx import CorrDiffTaiwan

        return CorrDiffTaiwan
    except ImportError:
        return None


def _try_import_cbottle():
    """Import cBottle model, returning None if unavailable."""
    try:
        from earth2studio.data import CBottle3D

        return CBottle3D
    except ImportError:
        return None


def downscale_corrdiff(
    coarse_data: xr.Dataset,
    time: datetime,
    num_samples: int = 1,
    num_steps: int = 8,
    inference_mode: str = "regression",
) -> xr.Dataset:
    """Downscale atmospheric data from 25km to ~3km using CorrDiff.

    CorrDiff combines a deterministic UNet regression with stochastic
    diffusion residuals for the Taiwan domain (448x448 grid).

    Args:
        coarse_data: ERA5/GFS dataset at 0.25° resolution.
        time: Target datetime.
        num_samples: Number of stochastic ensemble members.
        num_steps: Langevin diffusion steps (fewer = faster, less detail).
        inference_mode: 'regression' (fast), 'diffusion', or 'both'.

    Returns:
        xr.Dataset at ~3km resolution with variables:
        [u10m, v10m, t2m, tp, csnow, cicep, cfrzr, crain]
    """
    corrdiff_cls = _try_import_corrdiff()
    if corrdiff_cls is None:
        logger.warning("CorrDiff not available, returning bilinear upsampled data")
        return _bilinear_upsample(coarse_data, factor=8)

    logger.info(
        "Running CorrDiff downscaling: samples=%d, steps=%d, mode=%s",
        num_samples, num_steps, inference_mode,
    )

    try:
        import torch

        package = corrdiff_cls.load_default_package()
        model = corrdiff_cls.load_model(package)
        model.number_of_samples = num_samples
        model.number_of_steps = num_steps
        model.inference_mode = inference_mode

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("CorrDiff loaded on %s", device)

        # Run inference
        from earth2studio.data import GFS
        from earth2studio.io import KVBackend
        from earth2studio.run import diagnostic

        output = diagnostic(
            time=[time.isoformat()],
            diagnostic=model,
            data=GFS(),
            io=KVBackend(),
        )

        # Convert output to xr.Dataset
        output_vars = ["u10m", "v10m", "t2m", "tp", "csnow", "cicep", "cfrzr", "crain"]
        ds = xr.Dataset()
        for var in output_vars:
            if var in output:
                ds[var] = xr.DataArray(
                    output[var].squeeze(), dims=["lat", "lon"]
                )

        logger.info("CorrDiff output: %d variables at ~3km", len(ds.data_vars))

        # Free model weights to reclaim VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CorrDiff weights freed from VRAM")

        return ds

    except Exception:
        logger.exception("CorrDiff inference failed, falling back to bilinear")
        return _bilinear_upsample(coarse_data, factor=8)


def generate_cbottle(
    time: datetime,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Generate km-scale climate state using cBottle cascaded diffusion.

    Stage 1 (cBottle-3d): ~100km global on HPX64 grid (150M params)
    Stage 2 (cBottle-SR): 16x super-resolution to ~5km on HPX1024 (330M params)

    Args:
        time: Target datetime for climate state.
        variables: Output variables. Default: [msl, tcwv, t2m, u10m, v10m].

    Returns:
        xr.Dataset on HPX grid.
    """
    variables = variables or ["msl", "tcwv", "t2m", "u10m", "v10m"]

    cbottle_cls = _try_import_cbottle()
    if cbottle_cls is None:
        logger.warning("cBottle not available, generating synthetic climate state")
        return _generate_synthetic_climate(variables)

    logger.info("Running cBottle climate generation for %s", time)

    try:
        import torch

        package = cbottle_cls.load_default_package()
        model = cbottle_cls.load_model(package)

        if torch.cuda.is_available():
            model = model.to("cuda")

        da = model([time], variables)

        ds = xr.Dataset()
        for var in variables:
            var_data = da.sel(variable=var)
            if "variable" in var_data.dims:
                var_data = var_data.squeeze("variable", drop=True)
            if "time" in var_data.dims:
                var_data = var_data.squeeze("time", drop=True)
            ds[var] = var_data

        # Free VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("cBottle complete: %d variables", len(ds.data_vars))

        return ds

    except Exception:
        logger.exception("cBottle generation failed")
        return _generate_synthetic_climate(variables)


def _bilinear_upsample(ds: xr.Dataset, factor: int = 8) -> xr.Dataset:
    """Simple bilinear upsampling as fallback when AI models unavailable."""
    result = xr.Dataset()

    for var in ds.data_vars:
        data = ds[var].values
        if data.ndim < 2:
            result[var] = ds[var]
            continue

        from scipy.ndimage import zoom

        upsampled = zoom(data, factor, order=1).astype(np.float32)
        result[var] = xr.DataArray(upsampled, dims=ds[var].dims)

    # Update coordinates if they exist
    first_var = next(iter(result.data_vars))
    out_shape = result[first_var].shape
    if "lat" in ds.coords:
        lat = ds.coords["lat"].values
        result.coords["lat"] = np.linspace(lat[0], lat[-1], out_shape[0])
    if "lon" in ds.coords:
        lon = ds.coords["lon"].values
        result.coords["lon"] = np.linspace(lon[0], lon[-1], out_shape[1])

    logger.info("Bilinear upsample %dx: output shape %s", factor, out_shape)
    return result


def _generate_synthetic_climate(variables: list[str]) -> xr.Dataset:
    """Generate synthetic climate state for testing."""
    rng = np.random.default_rng(42)
    n = 64  # HPX64-like grid size

    ds = xr.Dataset()
    defaults: dict[str, Any] = {
        "msl": lambda: rng.normal(101325, 500, (n, n)).astype(np.float32),
        "tcwv": lambda: rng.uniform(5, 50, (n, n)).astype(np.float32),
        "t2m": lambda: rng.normal(288, 15, (n, n)).astype(np.float32),
        "u10m": lambda: rng.normal(0, 5, (n, n)).astype(np.float32),
        "v10m": lambda: rng.normal(0, 5, (n, n)).astype(np.float32),
    }

    for var in variables:
        gen = defaults.get(var, lambda: rng.normal(0, 1, (n, n)).astype(np.float32))
        ds[var] = xr.DataArray(gen(), dims=["y", "x"])

    return ds
