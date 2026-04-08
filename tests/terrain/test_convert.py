"""Tests for heightfield-to-Genesis converter."""

import numpy as np

from esimulab.terrain.convert import GenesisHeightfield, prepare_heightfield


class TestPrepareHeightfield:
    def test_basic_output(self):
        dem = np.ones((100, 200), dtype=np.float32) * 500.0
        result = prepare_heightfield(dem, pixel_size=30.0)

        assert isinstance(result, GenesisHeightfield)
        assert result.height_field.dtype == np.float32
        assert result.height_field.shape == (100, 200)
        assert result.horizontal_scale == 30.0
        assert result.vertical_scale == 1.0

    def test_nan_filled(self):
        dem = np.ones((50, 50), dtype=np.float32)
        dem[10, 10] = np.nan
        result = prepare_heightfield(dem, pixel_size=10.0, fill_nodata=-999.0)

        assert not np.any(np.isnan(result.height_field))
        assert result.height_field[10, 10] == -999.0

    def test_centered_origin(self):
        dem = np.zeros((100, 200), dtype=np.float32)
        result = prepare_heightfield(dem, pixel_size=30.0, center_origin=True)

        x_extent = 200 * 30.0
        y_extent = 100 * 30.0
        assert result.origin == (-x_extent / 2, -y_extent / 2, 0.0)

    def test_uncenterered_origin(self):
        dem = np.zeros((100, 200), dtype=np.float32)
        result = prepare_heightfield(dem, pixel_size=30.0, center_origin=False)

        assert result.origin == (0.0, 0.0, 0.0)

    def test_vertical_scale(self):
        dem = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        result = prepare_heightfield(dem, pixel_size=10.0, vertical_scale=2.0)

        assert result.vertical_scale == 2.0
        assert result.bounds_max[2] == 300.0 * 2.0
        assert result.bounds_min[2] == 0.0

    def test_bounds_computation(self):
        dem = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        result = prepare_heightfield(dem, pixel_size=5.0, center_origin=False)

        assert result.bounds_min == (0.0, 0.0, 10.0)
        assert result.bounds_max == (10.0, 10.0, 40.0)

    def test_preserves_data(self):
        dem = np.arange(25, dtype=np.float32).reshape(5, 5)
        result = prepare_heightfield(dem, pixel_size=1.0)

        np.testing.assert_array_equal(result.height_field, dem)
