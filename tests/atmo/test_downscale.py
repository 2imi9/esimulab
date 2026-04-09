"""Tests for AI downscaling (CorrDiff, cBottle)."""

from datetime import datetime

import numpy as np
import xarray as xr

from esimulab.atmo.downscale import (
    _bilinear_upsample,
    _generate_synthetic_climate,
    downscale_corrdiff,
    generate_cbottle,
)


class TestBilinearUpsample:
    def test_upsamples_by_factor(self):
        ds = xr.Dataset({
            "t2m": xr.DataArray(np.ones((4, 4), dtype=np.float32), dims=["lat", "lon"]),
        }, coords={"lat": np.arange(4), "lon": np.arange(4)})

        result = _bilinear_upsample(ds, factor=2)
        assert result["t2m"].shape == (8, 8)

    def test_preserves_values(self):
        data = np.full((4, 4), 42.0, dtype=np.float32)
        ds = xr.Dataset({
            "t2m": xr.DataArray(data, dims=["lat", "lon"]),
        }, coords={"lat": np.arange(4), "lon": np.arange(4)})

        result = _bilinear_upsample(ds, factor=3)
        np.testing.assert_allclose(result["t2m"].values, 42.0, atol=0.1)

    def test_updates_coordinates(self):
        ds = xr.Dataset({
            "t2m": xr.DataArray(np.ones((4, 4), dtype=np.float32), dims=["lat", "lon"]),
        }, coords={"lat": np.array([10, 20, 30, 40]), "lon": np.array([50, 60, 70, 80])})

        result = _bilinear_upsample(ds, factor=2)
        assert len(result.coords["lat"]) == 8
        assert result.coords["lat"].values[0] == 10
        assert result.coords["lat"].values[-1] == 40


class TestSyntheticClimate:
    def test_returns_requested_variables(self):
        ds = _generate_synthetic_climate(["t2m", "u10m", "msl"])
        assert "t2m" in ds
        assert "u10m" in ds
        assert "msl" in ds

    def test_shape_is_64x64(self):
        ds = _generate_synthetic_climate(["t2m"])
        assert ds["t2m"].shape == (64, 64)


class TestCorrDiffFallback:
    def test_falls_back_to_bilinear(self):
        """Without CorrDiff installed, should bilinear upsample."""
        coarse = xr.Dataset({
            "t2m": xr.DataArray(
                np.ones((4, 4), dtype=np.float32) * 290,
                dims=["lat", "lon"],
            ),
        }, coords={"lat": np.arange(4), "lon": np.arange(4)})

        result = downscale_corrdiff(coarse, datetime(2023, 6, 15))
        assert "t2m" in result
        assert result["t2m"].shape[0] > 4  # upsampled

    def test_custom_params(self):
        coarse = xr.Dataset({
            "u10m": xr.DataArray(np.ones((3, 3), dtype=np.float32), dims=["lat", "lon"]),
        }, coords={"lat": np.arange(3), "lon": np.arange(3)})

        result = downscale_corrdiff(
            coarse, datetime(2023, 1, 1),
            num_samples=2, num_steps=4, inference_mode="both",
        )
        assert "u10m" in result


class TestCBottleFallback:
    def test_falls_back_to_synthetic(self):
        result = generate_cbottle(datetime(2023, 6, 15))
        assert "t2m" in result
        assert "u10m" in result

    def test_custom_variables(self):
        result = generate_cbottle(datetime(2023, 6, 15), variables=["msl", "tcwv"])
        assert "msl" in result
        assert "tcwv" in result
        assert "t2m" not in result
