"""Tests for Skyfall-GS urban representation layer."""

import struct

import numpy as np

from esimulab.urban.skyfall_gs import (
    PRETRAINED_SCENES,
    SkyfallConfig,
    UrbanScene,
    check_availability,
    load_ply_metadata,
    ply_to_web_splat,
)


class TestSkyfallConfig:
    def test_defaults(self):
        config = SkyfallConfig()
        assert config.sh_degree == 1
        assert config.render_resolution == (1024, 1024)
        assert config.iterations_stage1 == 30000

    def test_pretrained_scenes(self):
        assert len(PRETRAINED_SCENES) >= 4
        assert "JAX_068" in PRETRAINED_SCENES
        assert "NYC_004" in PRETRAINED_SCENES


class TestCheckAvailability:
    def test_returns_status(self):
        status = check_availability()
        assert "available" in status
        assert "pretrained_scenes" in status
        assert isinstance(status["pretrained_scenes"], list)


class TestUrbanScene:
    def test_empty_scene(self):
        scene = UrbanScene()
        assert scene.n_gaussians == 0
        assert scene.ply_path is None


class TestPlyMetadata:
    def test_read_synthetic_ply(self, tmp_path):
        # Create a minimal PLY file
        ply_path = tmp_path / "test.ply"
        n = 100
        properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                       "scale_0", "scale_1", "scale_2", "opacity",
                       "rot_0", "rot_1", "rot_2", "rot_3"]
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for p in properties:
            header += f"property float {p}\n"
        header += "end_header\n"

        data = np.random.randn(n, len(properties)).astype(np.float32)
        with open(ply_path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())

        meta = load_ply_metadata(ply_path)
        assert meta["n_gaussians"] == 100
        assert len(meta["properties"]) == len(properties)


class TestPlyToSplat:
    def test_convert_synthetic(self, tmp_path):
        # Create synthetic PLY
        ply_path = tmp_path / "test.ply"
        n = 50
        properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                       "scale_0", "scale_1", "scale_2", "opacity",
                       "rot_0", "rot_1", "rot_2", "rot_3"]
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        for p in properties:
            header += f"property float {p}\n"
        header += "end_header\n"

        data = np.random.randn(n, len(properties)).astype(np.float32)
        with open(ply_path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())

        splat_path = ply_to_web_splat(ply_path, tmp_path / "test.splat")
        assert splat_path is not None
        assert splat_path.exists()

        # Verify .splat format: 32 bytes per Gaussian
        size = splat_path.stat().st_size
        assert size == n * 32

        # Read first Gaussian
        with open(splat_path, "rb") as f:
            x, y, z = struct.unpack("<fff", f.read(12))
            sx, sy, sz = struct.unpack("<fff", f.read(12))
            r, g, b, a = struct.unpack("BBBB", f.read(4))
            rr, rg, rb, ra = struct.unpack("BBBB", f.read(4))

        assert all(np.isfinite([x, y, z, sx, sy, sz]))
        assert 0 <= r <= 255

    def test_subsample(self, tmp_path):
        ply_path = tmp_path / "big.ply"
        n = 1000
        props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                 "scale_0", "scale_1", "scale_2", "opacity",
                 "rot_0", "rot_1", "rot_2", "rot_3"]
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        for p in props:
            header += f"property float {p}\n"
        header += "end_header\n"
        data = np.random.randn(n, len(props)).astype(np.float32)
        with open(ply_path, "wb") as f:
            f.write(header.encode())
            f.write(data.tobytes())

        splat = ply_to_web_splat(ply_path, tmp_path / "sub.splat", max_gaussians=100)
        assert splat.stat().st_size == 100 * 32
