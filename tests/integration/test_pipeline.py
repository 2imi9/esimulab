"""Integration test: full pipeline in data-only mode."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture_atmo():
    """Load sample atmosphere from JSON fixture into xr.Dataset."""
    data = json.loads((FIXTURES_DIR / "sample_atmo.json").read_text())
    lat = np.array(data["lat"])
    lon = np.array(data["lon"])
    ds = xr.Dataset(
        {
            "u10m": xr.DataArray(
                np.array(data["u10m"], dtype=np.float32), dims=["lat", "lon"]
            ),
            "v10m": xr.DataArray(
                np.array(data["v10m"], dtype=np.float32), dims=["lat", "lon"]
            ),
            "t2m": xr.DataArray(
                np.array(data["t2m"], dtype=np.float32), dims=["lat", "lon"]
            ),
            "tp": xr.DataArray(
                np.array(data["tp"], dtype=np.float32), dims=["lat", "lon"]
            ),
        },
        coords={"lat": lat, "lon": lon},
    )
    return ds


@pytest.mark.integration
def test_full_pipeline_data_only(tmp_path):
    """Run pipeline with mocked DEM fetch and real atmospheric extraction."""
    from esimulab.pipeline import run_pipeline

    # Mock DEM fetch to use fixture
    dem_data = np.load(FIXTURES_DIR / "sample_dem.npy")
    mock_dem_result = MagicMock()
    mock_dem_result.heightfield = dem_data
    mock_dem_result.pixel_size = 30.0

    # Mock ERA5 to use fixture
    atmo_ds = _load_fixture_atmo()

    with (
        patch("esimulab.terrain.fetch_dem", return_value=mock_dem_result),
        patch("esimulab.atmo.fetch_era5", return_value=atmo_ds),
    ):
        run_pipeline(
            bbox=(-119.0, 33.0, -118.0, 35.0),
            time=datetime(2023, 6, 15),
            output_dir=tmp_path,
            skip_gpu=True,
        )

    # Verify all output files exist
    assert (tmp_path / "terrain" / "heightfield.npy").exists()
    assert (tmp_path / "terrain" / "metadata.json").exists()
    assert (tmp_path / "atmo" / "wind.json").exists()
    assert (tmp_path / "atmo" / "precip.json").exists()
    assert (tmp_path / "metadata.json").exists()

    # Verify wind extraction produced real values
    wind = json.loads((tmp_path / "atmo" / "wind.json").read_text())
    assert wind["magnitude"] > 0
    assert len(wind["direction"]) == 3

    # Verify precip extraction
    precip = json.loads((tmp_path / "atmo" / "precip.json").read_text())
    assert precip["rate_mm_hr"] >= 0

    # Verify pipeline metadata
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["bbox"] == [-119.0, 33.0, -118.0, 35.0]
    assert meta["skip_gpu"] is True
