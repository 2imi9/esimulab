"""Tests for spatially-varying wind zones."""

from unittest.mock import MagicMock

import numpy as np

from esimulab.sim.wind_zones import WindZone, create_wind_zones, wind_zones_from_dataset


class TestCreateWindZones:
    def _make_hf(self):
        hf = MagicMock()
        hf.bounds_min = (-500.0, -500.0, 0.0)
        hf.bounds_max = (500.0, 500.0, 100.0)
        return hf

    def test_creates_zones_from_uniform_wind(self):
        u = np.full((8, 8), 5.0, dtype=np.float32)
        v = np.full((8, 8), 0.0, dtype=np.float32)
        zones = create_wind_zones(u, v, self._make_hf(), max_zones=9)
        assert len(zones) > 0
        assert all(isinstance(z, WindZone) for z in zones)

    def test_zones_have_correct_direction(self):
        u = np.full((4, 4), 3.0, dtype=np.float32)
        v = np.full((4, 4), 4.0, dtype=np.float32)
        zones = create_wind_zones(u, v, self._make_hf(), max_zones=4)
        for z in zones:
            assert abs(z.direction[0] - 0.6) < 0.1
            assert abs(z.direction[1] - 0.8) < 0.1

    def test_skips_calm_zones(self):
        u = np.zeros((4, 4), dtype=np.float32)
        v = np.zeros((4, 4), dtype=np.float32)
        zones = create_wind_zones(u, v, self._make_hf(), max_zones=4)
        assert len(zones) == 0

    def test_spatial_variation(self):
        u = np.zeros((4, 4), dtype=np.float32)
        u[:2, :] = 5.0  # north half windy
        v = np.zeros((4, 4), dtype=np.float32)
        zones = create_wind_zones(u, v, self._make_hf(), max_zones=4)
        strengths = [z.strength for z in zones]
        assert max(strengths) > 0


class TestWindZonesFromDataset:
    def test_from_dataset(self):
        import xarray as xr

        ds = xr.Dataset({
            "u10m": xr.DataArray(np.full((4, 4), 3.0, dtype=np.float32), dims=["lat", "lon"]),
            "v10m": xr.DataArray(np.full((4, 4), 4.0, dtype=np.float32), dims=["lat", "lon"]),
        })
        hf = MagicMock()
        hf.bounds_min = (-100.0, -100.0, 0.0)
        hf.bounds_max = (100.0, 100.0, 50.0)

        zones = wind_zones_from_dataset(ds, hf, max_zones=4)
        assert len(zones) > 0

    def test_from_scalar_dataset(self):
        import xarray as xr

        ds = xr.Dataset({
            "u10m": xr.DataArray(np.float32(5.0)),
            "v10m": xr.DataArray(np.float32(0.0)),
        })
        hf = MagicMock()
        hf.bounds_min = (0.0, 0.0, 0.0)
        hf.bounds_max = (100.0, 100.0, 50.0)

        zones = wind_zones_from_dataset(ds, hf, max_zones=1)
        assert len(zones) == 1
        assert zones[0].strength == 5.0
