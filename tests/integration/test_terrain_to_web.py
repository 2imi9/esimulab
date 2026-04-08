"""Integration test: terrain data -> web viewer API."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from esimulab.terrain.convert import prepare_heightfield
from esimulab.web.server import app

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def terrain_data(tmp_path):
    """Load sample DEM, convert, and save to tmp data dir."""
    dem = np.load(FIXTURES_DIR / "sample_dem.npy")
    hf = prepare_heightfield(dem, pixel_size=30.0)

    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir()
    np.save(terrain_dir / "heightfield.npy", hf.height_field)
    (terrain_dir / "metadata.json").write_text(json.dumps({
        "pixel_size": hf.horizontal_scale,
        "vertical_scale": hf.vertical_scale,
        "rows": hf.height_field.shape[0],
        "cols": hf.height_field.shape[1],
    }))
    return tmp_path


@pytest_asyncio.fixture
async def client(terrain_data, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", terrain_data)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terrain_roundtrip(client):
    """DEM -> prepare_heightfield -> save -> API -> binary parse."""
    resp = await client.get("/api/terrain")
    assert resp.status_code == 200

    data = resp.content
    rows, cols = struct.unpack("<II", data[:8])
    assert rows == 20
    assert cols == 20

    values = np.frombuffer(data[8:], dtype=np.float32).reshape(rows, cols)
    assert not np.any(np.isnan(values))
    assert values.min() >= 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terrain_metadata_roundtrip(client):
    resp = await client.get("/api/terrain/metadata")
    assert resp.status_code == 200
    meta = resp.json()
    assert meta["pixel_size"] == 30.0
    assert meta["rows"] == 20
    assert meta["cols"] == 20
