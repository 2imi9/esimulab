"""Tests for the FastAPI web viewer server."""

from __future__ import annotations

import json
import struct

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
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_terrain_404_when_no_data(client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    resp = await client.get("/api/terrain")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_terrain_returns_binary(client, tmp_path, monkeypatch):
    terrain_dir = tmp_path / "terrain"
    terrain_dir.mkdir()

    hf = np.ones((10, 20), dtype=np.float32) * 500
    np.save(terrain_dir / "heightfield.npy", hf)

    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)

    resp = await client.get("/api/terrain")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    data = resp.content
    rows, cols = struct.unpack("<II", data[:8])
    assert rows == 10
    assert cols == 20

    values = np.frombuffer(data[8:], dtype=np.float32).reshape(rows, cols)
    np.testing.assert_array_equal(values, hf)


@pytest.mark.asyncio
async def test_frames_empty_when_no_data(client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    resp = await client.get("/api/frames")
    assert resp.status_code == 200
    assert resp.json()["frames"] == []


@pytest.mark.asyncio
async def test_frames_lists_available(client, tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000000.bin").write_bytes(b"\x00" * 16)
    (frames_dir / "frame_000010.bin").write_bytes(b"\x00" * 16)

    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)

    resp = await client.get("/api/frames")
    data = resp.json()
    assert data["count"] == 2
    assert "frame_000000" in data["frames"]


@pytest.mark.asyncio
async def test_frame_404(client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    resp = await client.get("/api/frames/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_metadata_no_simulation(client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    resp = await client.get("/api/metadata")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_simulation"


@pytest.mark.asyncio
async def test_metadata_returns_json(client, tmp_path, monkeypatch):
    meta = {"steps": 1000, "dt": 0.002}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))

    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)

    resp = await client.get("/api/metadata")
    assert resp.status_code == 200
    assert resp.json()["steps"] == 1000
