"""Tests for renderer selection and RayTracer config."""

from unittest.mock import MagicMock

from esimulab.sim.renderer import (
    RayTracerConfig,
    check_luisa_render,
    create_renderer,
)


class TestRayTracerConfig:
    def test_defaults(self):
        config = RayTracerConfig()
        assert config.tracing_depth == 32
        assert config.spp == 64
        assert config.spp_final == 256

    def test_default_lights(self):
        config = RayTracerConfig()
        lights = config.default_lights(z_top=300)
        assert len(lights) == 2
        assert lights[0]["pos"][2] > 300


class TestCreateRenderer:
    def test_force_rasterizer(self):
        gs = MagicMock()
        renderer, name = create_renderer(gs, force_rasterizer=True)
        assert name == "rasterizer"
        gs.renderers.Rasterizer.assert_called_once()

    def test_non_linux_gets_rasterizer(self):
        gs = MagicMock()
        renderer, name = create_renderer(gs)
        # On Windows (where tests run), should get rasterizer
        assert name == "rasterizer"


class TestCheckLuisaRender:
    def test_returns_dict(self):
        info = check_luisa_render()
        assert "available" in info
        assert "platform" in info
        assert "reason" in info
        assert isinstance(info["reason"], str)
