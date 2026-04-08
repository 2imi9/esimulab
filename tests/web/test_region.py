"""Tests for region selection API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from esimulab.web.server import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_region_invalid_bounds(client):
    resp = await client.post(
        "/api/region",
        json={"west": 10, "south": 50, "east": 5, "north": 40},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_region_too_large(client):
    resp = await client.post(
        "/api/region",
        json={"west": -120, "south": 30, "east": -110, "north": 40},
    )
    assert resp.status_code == 400
    assert "too large" in resp.json()["detail"].lower()


@pytest.mark.asyncio
@patch("esimulab.terrain.fetch_dem")
@patch("esimulab.terrain.prepare_heightfield")
async def test_region_success(mock_hf, mock_dem, client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)

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

    resp = await client.post(
        "/api/region",
        json={"west": -119.1, "south": 34.0, "east": -119.0, "north": 34.1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["metadata"]["rows"] == 10
    assert (tmp_path / "terrain" / "heightfield.npy").exists()


@pytest.mark.asyncio
async def test_globe_page_loads(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "cesium" in resp.text.lower() or "Esimulab" in resp.text


@pytest.mark.asyncio
async def test_viewer_page_loads(client):
    resp = await client.get("/viewer")
    assert resp.status_code == 200
    assert "three" in resp.text.lower() or "canvas" in resp.text.lower()
