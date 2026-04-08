"""Tests for atmospheric data fetching."""

from datetime import datetime

import numpy as np

from esimulab.atmo.fetch import _generate_synthetic, fetch_era5


class TestGenerateSynthetic:
    def test_returns_dataset_with_variables(self):
        ds = _generate_synthetic(["u10m", "v10m", "t2m"], -119.0, 33.0, -118.0, 35.0)

        assert "u10m" in ds
        assert "v10m" in ds
        assert "t2m" in ds
        assert "lat" in ds.coords
        assert "lon" in ds.coords

    def test_grid_covers_bbox(self):
        ds = _generate_synthetic(["u10m"], -120.0, 34.0, -118.0, 36.0)

        assert ds.coords["lat"].values[0] >= 34.0
        assert ds.coords["lat"].values[-1] <= 36.0
        assert ds.coords["lon"].values[0] >= -120.0
        assert ds.coords["lon"].values[-1] <= -118.0

    def test_values_are_float32(self):
        ds = _generate_synthetic(["u10m", "t2m"], -119.0, 33.0, -118.0, 35.0)

        assert ds["u10m"].dtype == np.float32
        assert ds["t2m"].dtype == np.float32

    def test_deterministic_with_seed(self):
        ds1 = _generate_synthetic(["u10m"], -119.0, 33.0, -118.0, 35.0)
        ds2 = _generate_synthetic(["u10m"], -119.0, 33.0, -118.0, 35.0)

        np.testing.assert_array_equal(ds1["u10m"].values, ds2["u10m"].values)


class TestFetchEra5:
    def test_synthetic_fallback(self):
        """Without Earth2Studio installed, should return synthetic data."""
        bbox = (-119.1, 33.4, -118.9, 35.4)
        time = datetime(2023, 6, 15)

        ds = fetch_era5(bbox, time)

        assert "u10m" in ds
        assert "v10m" in ds
        assert "t2m" in ds
        assert len(ds.coords["lat"]) > 0
        assert len(ds.coords["lon"]) > 0

    def test_custom_variables(self):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        time = datetime(2023, 6, 15)

        ds = fetch_era5(bbox, time, variables=["u10m", "v10m"])

        assert "u10m" in ds
        assert "v10m" in ds
        assert "t2m" not in ds
