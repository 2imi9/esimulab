"""Tests for DEM fetching and reprojection."""

from unittest.mock import patch

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from esimulab.terrain.dem import DEMResult, _estimate_utm_zone, fetch_dem


class TestEstimateUtmZone:
    def test_northern_hemisphere(self):
        crs = _estimate_utm_zone((-122.5, 37.0, -122.0, 37.5))
        assert crs.to_epsg() == 32610  # UTM 10N (San Francisco)

    def test_southern_hemisphere(self):
        crs = _estimate_utm_zone((151.0, -34.0, 151.5, -33.5))
        assert crs.to_epsg() == 32756  # UTM 56S (Sydney)

    def test_prime_meridian(self):
        crs = _estimate_utm_zone((-1.0, 51.0, 1.0, 52.0))
        assert crs.to_epsg() == 32631  # UTM 31N (London area)


class TestFetchDem:
    def _make_mock_profile(self, bbox, shape=(100, 100)):
        west, south, east, north = bbox
        transform = from_bounds(west, south, east, north, shape[1], shape[0])
        return {
            "crs": CRS.from_epsg(4326),
            "transform": transform,
            "width": shape[1],
            "height": shape[0],
        }

    @patch("dem_stitcher.stitch_dem")
    def test_returns_dem_result(self, mock_stitch):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        fake_data = np.random.rand(100, 100).astype(np.float32) * 1000
        mock_stitch.return_value = (fake_data, self._make_mock_profile(bbox))

        result = fetch_dem(bbox, reproject_to_utm=True)

        assert isinstance(result, DEMResult)
        assert result.heightfield.dtype == np.float32
        assert result.pixel_size > 0
        assert result.bounds == bbox
        mock_stitch.assert_called_once()

    @patch("dem_stitcher.stitch_dem")
    def test_no_reproject(self, mock_stitch):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        fake_data = np.random.rand(50, 50).astype(np.float32)
        mock_stitch.return_value = (fake_data, self._make_mock_profile(bbox, (50, 50)))

        result = fetch_dem(bbox, reproject_to_utm=False)

        assert result.crs.to_epsg() == 4326
        assert result.heightfield.shape == (50, 50)

    @patch("dem_stitcher.stitch_dem")
    def test_handles_nodata_nan(self, mock_stitch):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        fake_data = np.ones((50, 50), dtype=np.float32)
        fake_data[10:20, 10:20] = np.nan
        mock_stitch.return_value = (fake_data, self._make_mock_profile(bbox, (50, 50)))

        result = fetch_dem(bbox, reproject_to_utm=False)

        assert np.isnan(result.heightfield[15, 15])

    @patch("dem_stitcher.stitch_dem")
    def test_dem_name_passed_through(self, mock_stitch):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        fake_data = np.zeros((10, 10), dtype=np.float32)
        mock_stitch.return_value = (fake_data, self._make_mock_profile(bbox, (10, 10)))

        fetch_dem(bbox, dem_name="cop_30", reproject_to_utm=False)

        _, kwargs = mock_stitch.call_args
        assert kwargs["dem_name"] == "cop_30"
