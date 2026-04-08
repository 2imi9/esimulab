"""Tests for land cover fetching and alignment."""

from unittest.mock import MagicMock, patch

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from esimulab.terrain.landcover import _tile_urls_for_bbox, fetch_landcover


class TestTileUrls:
    def test_single_tile(self):
        urls = _tile_urls_for_bbox(-119.0, 34.0, -118.0, 35.0)
        assert len(urls) == 1
        assert "N33" in urls[0] or "N34" in urls[0]

    def test_multi_tile_crossing_boundary(self):
        urls = _tile_urls_for_bbox(-121.0, 33.0, -118.0, 37.0)
        assert len(urls) >= 2

    def test_southern_hemisphere(self):
        urls = _tile_urls_for_bbox(151.0, -34.0, 152.0, -33.0)
        assert any("S" in u for u in urls)


class TestFetchLandcover:
    def _make_dem_profile(self, bbox, shape=(50, 50)):
        west, south, east, north = bbox
        return {
            "crs": CRS.from_epsg(32611),
            "transform": from_bounds(0, 0, shape[1] * 30, shape[0] * 30, shape[1], shape[0]),
            "width": shape[1],
            "height": shape[0],
        }

    @patch("esimulab.terrain.landcover.rioxarray")
    def test_returns_zeros_on_no_tiles(self, mock_rio):
        mock_rio.open_rasterio.side_effect = Exception("no tile")
        bbox = (-119.1, 33.4, -118.9, 35.4)
        profile = self._make_dem_profile(bbox)

        result = fetch_landcover(bbox, profile)

        assert result.dtype == np.uint8
        assert result.shape == (50, 50)
        assert np.all(result == 0)

    @patch("esimulab.terrain.landcover.rioxarray")
    def test_returns_aligned_array(self, mock_rio):
        bbox = (-119.1, 33.4, -118.9, 35.4)
        profile = self._make_dem_profile(bbox, shape=(30, 30))

        # Create a mock tile with rio accessor
        mock_tile = MagicMock()
        mock_clip = MagicMock()
        mock_reprojected = MagicMock()

        mock_rio.open_rasterio.return_value = mock_tile
        mock_tile.rio.clip_box.return_value = mock_clip
        mock_clip.rio.reproject.return_value = mock_reprojected
        mock_reprojected.values.squeeze.return_value = np.full(
            (30, 30), 40, dtype=np.uint8
        )

        result = fetch_landcover(bbox, profile)

        assert result.shape == (30, 30)
        assert result.dtype == np.uint8
        assert np.all(result == 40)  # cropland class
