"""Tests for parallel simulation and differentiable mode."""

from esimulab.sim.parallel import DiffConfig, ParallelConfig, setup_differentiable


class TestParallelConfig:
    def test_defaults(self):
        config = ParallelConfig()
        assert config.n_envs == 1
        assert config.vary_wind is True

    def test_custom(self):
        config = ParallelConfig(n_envs=100, vary_terrain=True)
        assert config.n_envs == 100


class TestDiffConfig:
    def test_defaults(self):
        config = DiffConfig()
        assert config.requires_grad is False

    def test_enable_grad(self):
        config = DiffConfig(requires_grad=True, optimize_target="wind")
        assert config.optimize_target == "wind"


class TestSetupDifferentiable:
    def test_enables_grad(self):
        import unittest.mock as mock

        gs = mock.MagicMock()
        kwargs = {"dt": 0.01, "substeps": 4}
        config = DiffConfig(requires_grad=True)
        result = setup_differentiable(gs, kwargs, config)
        assert result["requires_grad"] is True

    def test_disabled_by_default(self):
        import unittest.mock as mock

        gs = mock.MagicMock()
        kwargs = {"dt": 0.01}
        result = setup_differentiable(gs, kwargs)
        assert result["requires_grad"] is False
