"""Tests for precipitation extraction."""

import numpy as np
import xarray as xr

from esimulab.atmo.precip import PrecipForcing, extract_precip_rate


class TestExtractPrecipRate:
    def _make_ds(self, precip_val: float, shape: tuple = (8, 8)) -> xr.Dataset:
        lat = np.linspace(33, 35, shape[0])
        lon = np.linspace(-119, -118, shape[1])
        return xr.Dataset({
            "tp": xr.DataArray(
                np.full(shape, precip_val, dtype=np.float32), dims=["lat", "lon"]
            ),
        }, coords={"lat": lat, "lon": lon})

    def test_returns_precip_forcing(self):
        ds = self._make_ds(5.0)
        result = extract_precip_rate(ds)

        assert isinstance(result, PrecipForcing)
        assert result.rate_mm_hr == 5.0

    def test_zero_precip(self):
        ds = self._make_ds(0.0)
        result = extract_precip_rate(ds)

        assert result.rate_mm_hr == 0.0

    def test_missing_variable_returns_zero(self):
        ds = xr.Dataset({"other_var": xr.DataArray([1.0])})
        result = extract_precip_rate(ds)

        assert result.rate_mm_hr == 0.0

    def test_droplet_size_scales_with_intensity(self):
        light = extract_precip_rate(self._make_ds(1.0))
        heavy = extract_precip_rate(self._make_ds(20.0))

        assert heavy.droplet_size > light.droplet_size

    def test_negative_precip_clamped_to_zero(self):
        ds = self._make_ds(-5.0)
        result = extract_precip_rate(ds)

        assert result.rate_mm_hr == 0.0

    def test_terminal_velocity_default(self):
        ds = self._make_ds(5.0)
        result = extract_precip_rate(ds)

        assert result.terminal_velocity == 9.0
