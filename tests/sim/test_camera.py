"""Tests for camera and multi-modal rendering."""

import numpy as np

from esimulab.sim.camera import CameraConfig, RenderOutput, save_render


class TestCameraConfig:
    def test_defaults(self):
        config = CameraConfig()
        assert config.resolution == (1920, 1080)
        assert config.fov == 45.0
        assert config.model == "pinhole"

    def test_custom(self):
        config = CameraConfig(resolution=(1280, 720), spp=256, model="thinlens")
        assert config.spp == 256


class TestRenderOutput:
    def test_empty_output(self):
        output = RenderOutput()
        assert output.rgb is None
        assert output.depth is None

    def test_with_data(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)
        output = RenderOutput(rgb=rgb, depth=depth)
        assert output.rgb.shape == (100, 100, 3)
        assert output.depth.shape == (100, 100)


class TestSaveRender:
    def test_save_rgb(self, tmp_path):
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        output = RenderOutput(rgb=rgb)
        try:
            saved = save_render(output, tmp_path, frame_id="test")
            assert len(saved) >= 1
        except ImportError:
            # PIL not available — saves as .npy instead
            saved = save_render(output, tmp_path, frame_id="test")
            assert any(str(p).endswith(".npy") for p in saved)
