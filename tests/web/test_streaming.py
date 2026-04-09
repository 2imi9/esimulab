"""Tests for WebSocket streaming and particle subsampling."""

import struct

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from esimulab.web.server import app
from esimulab.web.streaming import load_and_subsample_frame, subsample_particles


class TestSubsampleParticles:
    def test_no_subsample_when_under_limit(self):
        pos = np.random.rand(100, 3).astype(np.float32)
        result = subsample_particles(pos, max_count=200)
        assert result.shape == (100, 3)

    def test_subsamples_when_over_limit(self):
        pos = np.random.rand(100000, 3).astype(np.float32)
        result = subsample_particles(pos, max_count=5000)
        assert result.shape == (5000, 3)

    def test_deterministic(self):
        pos = np.random.rand(10000, 3).astype(np.float32)
        r1 = subsample_particles(pos, max_count=100)
        r2 = subsample_particles(pos, max_count=100)
        np.testing.assert_array_equal(r1, r2)


class TestLoadAndSubsample:
    def test_load_valid_frame(self, tmp_path):
        pos = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        frame_path = tmp_path / "frame.bin"
        with open(frame_path, "wb") as f:
            f.write(struct.pack("<I", 2))
            f.write(pos.tobytes())

        result = load_and_subsample_frame(frame_path, max_particles=100)
        assert result is not None
        np.testing.assert_array_equal(result, pos)

    def test_returns_none_for_missing(self, tmp_path):
        result = load_and_subsample_frame(tmp_path / "nonexistent.bin")
        assert result is None


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_subsampled_endpoint(client, tmp_path, monkeypatch):
    import esimulab.web.server as srv
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    pos = np.random.rand(1000, 3).astype(np.float32)
    with open(frames_dir / "frame_000000.bin", "wb") as f:
        f.write(struct.pack("<I", 1000))
        f.write(pos.tobytes())

    resp = await client.get("/api/frames/frame_000000/subsampled?max_particles=100")
    assert resp.status_code == 200

    data = resp.content
    n = struct.unpack("<I", data[:4])[0]
    assert n == 100  # subsampled
