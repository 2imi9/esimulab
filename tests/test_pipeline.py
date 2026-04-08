"""Tests for pipeline orchestration."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np


class TestRunPipeline:
    @patch("esimulab.terrain.fetch_dem")
    @patch("esimulab.terrain.prepare_heightfield")
    @patch("esimulab.atmo.fetch_era5")
    @patch("esimulab.atmo.extract_wind_forcing")
    @patch("esimulab.atmo.extract_precip_rate")
    def test_data_only_mode(
        self, mock_precip, mock_wind, mock_era5, mock_hf, mock_dem, tmp_path
    ):
        """Pipeline with skip_gpu=True should fetch data but not simulate."""
        from esimulab.pipeline import run_pipeline

        # Setup mocks
        mock_dem_result = MagicMock()
        mock_dem_result.heightfield = np.ones((10, 10), dtype=np.float32)
        mock_dem_result.pixel_size = 30.0
        mock_dem.return_value = mock_dem_result

        mock_hf_result = MagicMock()
        mock_hf_result.height_field = np.ones((10, 10), dtype=np.float32)
        mock_hf_result.horizontal_scale = 30.0
        mock_hf_result.vertical_scale = 1.0
        mock_hf_result.origin = (0.0, 0.0, 0.0)
        mock_hf_result.bounds_min = (0.0, 0.0, 0.0)
        mock_hf_result.bounds_max = (300.0, 300.0, 1.0)
        mock_hf.return_value = mock_hf_result

        mock_wind_result = MagicMock()
        mock_wind_result.direction = (1.0, 0.0, 0.0)
        mock_wind_result.magnitude = 5.0
        mock_wind_result.turbulence_strength = 1.0
        mock_wind.return_value = mock_wind_result

        mock_precip_result = MagicMock()
        mock_precip_result.rate_mm_hr = 3.0
        mock_precip_result.terminal_velocity = 9.0
        mock_precip_result.droplet_size = 0.05
        mock_precip.return_value = mock_precip_result

        run_pipeline(
            bbox=(-119.1, 33.4, -118.9, 35.4),
            time=datetime(2023, 6, 15),
            output_dir=tmp_path,
            skip_gpu=True,
        )

        # Verify terrain saved
        assert (tmp_path / "terrain" / "heightfield.npy").exists()
        assert (tmp_path / "terrain" / "metadata.json").exists()

        # Verify atmosphere saved
        assert (tmp_path / "atmo" / "wind.json").exists()
        assert (tmp_path / "atmo" / "precip.json").exists()

        # Verify pipeline metadata
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["skip_gpu"] is True
        assert meta["steps"] == 600

    @patch("esimulab.terrain.fetch_dem")
    @patch("esimulab.terrain.prepare_heightfield")
    @patch("esimulab.atmo.fetch_era5")
    @patch("esimulab.atmo.extract_wind_forcing")
    @patch("esimulab.atmo.extract_precip_rate")
    def test_terrain_metadata_format(
        self, mock_precip, mock_wind, mock_era5, mock_hf, mock_dem, tmp_path
    ):
        from esimulab.pipeline import run_pipeline

        mock_dem.return_value = MagicMock(
            heightfield=np.zeros((5, 8), dtype=np.float32), pixel_size=10.0
        )
        mock_hf_result = MagicMock()
        mock_hf_result.height_field = np.zeros((5, 8), dtype=np.float32)
        mock_hf_result.horizontal_scale = 10.0
        mock_hf_result.vertical_scale = 1.0
        mock_hf_result.origin = (0.0, 0.0, 0.0)
        mock_hf_result.bounds_min = (0.0, 0.0, 0.0)
        mock_hf_result.bounds_max = (80.0, 50.0, 0.0)
        mock_hf.return_value = mock_hf_result
        mock_wind.return_value = MagicMock(
            direction=(1, 0, 0), magnitude=0, turbulence_strength=0
        )
        mock_precip.return_value = MagicMock(
            rate_mm_hr=0, terminal_velocity=9, droplet_size=0.05
        )

        run_pipeline(
            bbox=(-1, -1, 1, 1),
            time=datetime(2023, 1, 1),
            output_dir=tmp_path,
            skip_gpu=True,
        )

        meta = json.loads((tmp_path / "terrain" / "metadata.json").read_text())
        assert meta["rows"] == 5
        assert meta["cols"] == 8
        assert meta["pixel_size"] == 10.0
