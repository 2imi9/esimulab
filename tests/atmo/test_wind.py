"""Tests for wind field extraction."""

import numpy as np
import xarray as xr

from esimulab.atmo.wind import WindForcing, extract_wind_forcing


class TestExtractWindForcing:
    def _make_ds(self, u_val: float, v_val: float, shape: tuple = (8, 8)) -> xr.Dataset:
        lat = np.linspace(33, 35, shape[0])
        lon = np.linspace(-119, -118, shape[1])
        return xr.Dataset({
            "u10m": xr.DataArray(np.full(shape, u_val, dtype=np.float32), dims=["lat", "lon"]),
            "v10m": xr.DataArray(np.full(shape, v_val, dtype=np.float32), dims=["lat", "lon"]),
        }, coords={"lat": lat, "lon": lon})

    def test_returns_wind_forcing(self):
        ds = self._make_ds(3.0, 4.0)
        result = extract_wind_forcing(ds)

        assert isinstance(result, WindForcing)
        assert result.magnitude > 0

    def test_eastward_wind(self):
        ds = self._make_ds(5.0, 0.0)
        result = extract_wind_forcing(ds)

        assert abs(result.direction[0] - 1.0) < 1e-5
        assert abs(result.direction[1]) < 1e-5
        assert abs(result.magnitude - 5.0) < 1e-5

    def test_combined_wind(self):
        ds = self._make_ds(3.0, 4.0)
        result = extract_wind_forcing(ds)

        assert abs(result.magnitude - 5.0) < 1e-5
        assert abs(result.direction[0] - 0.6) < 1e-5
        assert abs(result.direction[1] - 0.8) < 1e-5

    def test_zero_wind(self):
        ds = self._make_ds(0.0, 0.0)
        result = extract_wind_forcing(ds)

        assert result.magnitude == 0.0
        assert result.direction == (1.0, 0.0, 0.0)  # default

    def test_turbulence_fraction(self):
        ds = self._make_ds(10.0, 0.0)
        result = extract_wind_forcing(ds, turbulence_fraction=0.3)

        assert abs(result.turbulence_strength - 3.0) < 1e-5

    def test_direction_2d_property(self):
        ds = self._make_ds(3.0, 4.0)
        result = extract_wind_forcing(ds)

        assert len(result.direction_2d) == 2
        assert result.direction_2d == (result.direction[0], result.direction[1])
