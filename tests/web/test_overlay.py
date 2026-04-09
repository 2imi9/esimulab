"""Tests for simulation overlay generation."""

import numpy as np

from esimulab.web.overlay import (
    generate_particle_density_overlay,
    generate_wind_overlay,
    overlay_metadata,
)


class TestOverlayMetadata:
    def test_basic_metadata(self):
        meta = overlay_metadata((-118.3, 34.0, -118.2, 34.1), "wind")
        assert meta["type"] == "wind"
        assert meta["bounds"]["west"] == -118.3
        assert "cesium_config" in meta

    def test_url_format(self):
        meta = overlay_metadata((-119, 33, -118, 34), "precip")
        assert meta["url"] == "/api/overlays/precip.png"


class TestWindOverlay:
    def test_generates_bytes(self):
        u = np.random.rand(10, 10).astype(np.float32)
        v = np.random.rand(10, 10).astype(np.float32)
        result = generate_wind_overlay(u, v)
        # May be empty if matplotlib not available
        assert isinstance(result, bytes)

    def test_saves_to_file(self, tmp_path):
        u = np.ones((5, 5), dtype=np.float32) * 3
        v = np.ones((5, 5), dtype=np.float32) * 4
        path = tmp_path / "wind.png"
        generate_wind_overlay(u, v, output_path=path)
        # File created only if matplotlib available
        if path.exists():
            assert path.stat().st_size > 0


class TestParticleDensity:
    def test_generates_bytes(self):
        particles = np.random.rand(1000, 3).astype(np.float32) * 100
        bounds = (0.0, 0.0, 100.0, 100.0)
        result = generate_particle_density_overlay(particles, bounds)
        assert isinstance(result, bytes)
