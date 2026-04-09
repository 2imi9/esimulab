"""Tests for GaussianCity integration.

Tests the data preparation and availability checking (no Docker/CUDA needed).
Actual inference requires Docker with CUDA 11.8.
"""

import json

import numpy as np

from esimulab.urban.gaussian_city import (
    GaussianCityConfig,
    check_availability,
    prepare_osm_projections,
)


class TestGaussianCityConfig:
    def test_defaults(self):
        config = GaussianCityConfig()
        assert config.n_frames == 24
        assert config.sensor_size == (960, 540)
        assert config.use_docker is True


class TestCheckAvailability:
    def test_returns_status_dict(self):
        status = check_availability()
        assert "available" in status
        assert "reason" in status
        assert "docker_available" in status
        assert "models_downloaded" in status
        assert isinstance(status["reason"], str)

    def test_models_not_downloaded(self):
        status = check_availability()
        # Models unlikely to be downloaded in test env
        # But the function should not crash
        assert isinstance(status["models_downloaded"], bool)


class TestPrepareProjections:
    def test_creates_projection_files(self, tmp_path):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        output_dir = str(tmp_path / "projections")

        result = prepare_osm_projections(bbox, output_dir)

        if result is not None:
            # Check projection files created
            proj_dir = result / "Projection"
            assert (proj_dir / "SEG.npy").exists()
            assert (proj_dir / "TD_HF.npy").exists()
            assert (result / "CENTERS.pkl").exists()
            assert (result / "metadata.json").exists()

            # Check SEG map
            seg = np.load(proj_dir / "SEG.npy")
            assert seg.shape == (2048, 2048)
            assert seg.dtype == np.uint8
            # Should have building classes (2=facade, 3=roof)
            assert np.any(seg == 2) or np.any(seg == 3)

            # Check height field
            hf = np.load(proj_dir / "TD_HF.npy")
            assert hf.shape == (2048, 2048)
            assert hf.max() > 0  # some building heights

            # Check metadata
            meta = json.loads((result / "metadata.json").read_text())
            assert meta["bbox"] == list(bbox)
            assert meta["building_count"] > 0

    def test_empty_bbox_handles_gracefully(self, tmp_path):
        # Very small bbox in ocean — no buildings
        bbox = (0.0, 0.0, 0.001, 0.001)
        # Should not crash — may return None or a dir with zero buildings
        prepare_osm_projections(bbox, str(tmp_path / "empty"))


class TestProjectionFormat:
    """Verify our projections match GaussianCity's expected format."""

    def test_seg_classes_valid(self, tmp_path):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        result = prepare_osm_projections(bbox, str(tmp_path / "fmt"))
        if result is None:
            return

        seg = np.load(result / "Projection" / "SEG.npy")
        unique_classes = np.unique(seg)
        # All classes should be 0-7 (GaussianCity has 8 classes)
        assert all(0 <= c <= 7 for c in unique_classes)

    def test_centers_format(self, tmp_path):
        import pickle

        bbox = (-118.3, 34.0, -118.2, 34.1)
        result = prepare_osm_projections(bbox, str(tmp_path / "ctr"))
        if result is None:
            return

        with open(result / "CENTERS.pkl", "rb") as f:
            centers = pickle.load(f)

        assert isinstance(centers, dict)
        if centers:
            # Each center should be (cx, cy, w, h, d)
            first_key = next(iter(centers))
            assert isinstance(first_key, int)
            val = centers[first_key]
            assert len(val) == 5
