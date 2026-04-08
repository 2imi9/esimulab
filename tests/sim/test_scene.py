"""Tests for Genesis scene builder (mocked — no GPU required)."""

from unittest.mock import MagicMock, patch

import numpy as np

from esimulab.atmo.precip import PrecipForcing
from esimulab.atmo.wind import WindForcing
from esimulab.terrain.convert import GenesisHeightfield


def _make_heightfield():
    return GenesisHeightfield(
        height_field=np.ones((50, 50), dtype=np.float32) * 100,
        horizontal_scale=30.0,
        vertical_scale=1.0,
        origin=(-750.0, -750.0, 0.0),
        bounds_min=(-750.0, -750.0, 100.0),
        bounds_max=(750.0, 750.0, 100.0),
    )


def _make_wind():
    return WindForcing(
        direction=(1.0, 0.0, 0.0),
        magnitude=5.0,
        turbulence_strength=1.0,
        turbulence_frequency=5,
    )


def _make_precip():
    return PrecipForcing(
        rate_mm_hr=5.0,
        terminal_velocity=9.0,
        droplet_size=0.05,
    )


class TestBuildScene:
    @patch("esimulab.sim.scene._import_genesis")
    def test_builds_scene_with_all_components(self, mock_import):
        gs = MagicMock()
        mock_import.return_value = gs
        gs.gpu = "gpu"

        from esimulab.sim.scene import build_scene

        result = build_scene(
            heightfield=_make_heightfield(),
            wind=_make_wind(),
            precip=_make_precip(),
        )

        assert "scene" in result
        assert "terrain" in result
        assert "emitter" in result
        assert "camera" in result

        gs.init.assert_called_once_with(backend="gpu", precision="32")
        gs.Scene.assert_called_once()

    @patch("esimulab.sim.scene._import_genesis")
    def test_builds_scene_without_wind(self, mock_import):
        gs = MagicMock()
        mock_import.return_value = gs
        gs.gpu = "gpu"

        from esimulab.sim.scene import build_scene

        build_scene(heightfield=_make_heightfield(), wind=None, precip=None)

        # SF options should not be in scene kwargs
        call_kwargs = gs.Scene.call_args[1]
        assert "sf_options" not in call_kwargs

    @patch("esimulab.sim.scene._import_genesis")
    def test_no_emitter_without_precip(self, mock_import):
        gs = MagicMock()
        mock_import.return_value = gs
        gs.gpu = "gpu"

        from esimulab.sim.scene import build_scene

        result = build_scene(
            heightfield=_make_heightfield(), wind=None, precip=None
        )

        assert result["emitter"] is None

    @patch("esimulab.sim.scene._import_genesis")
    def test_terrain_morph_uses_heightfield(self, mock_import):
        gs = MagicMock()
        mock_import.return_value = gs
        gs.gpu = "gpu"

        from esimulab.sim.scene import build_scene

        hf = _make_heightfield()
        build_scene(heightfield=hf)

        gs.morphs.Terrain.assert_called_once()
        call_kwargs = gs.morphs.Terrain.call_args[1]
        np.testing.assert_array_equal(call_kwargs["height_field"], hf.height_field)
        assert call_kwargs["horizontal_scale"] == 30.0


class TestRunnerExport:
    def test_export_frame_binary_format(self, tmp_path):
        """Test the binary frame export format."""
        from esimulab.sim.runner import _export_frame

        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out_path = tmp_path / "frame.bin"
        _export_frame(positions, out_path)

        with open(out_path, "rb") as f:
            import struct

            (n,) = struct.unpack("<I", f.read(4))
            data = np.frombuffer(f.read(), dtype=np.float32).reshape(n, 3)

        assert n == 2
        np.testing.assert_array_equal(data, positions)
