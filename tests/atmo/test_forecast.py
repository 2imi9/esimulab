"""Tests for weather forecast integration."""

from datetime import datetime

from esimulab.atmo.forecast import generate_synthetic_forecast, run_forecast


class TestRunForecast:
    def test_returns_none_without_model(self):
        result = run_forecast(datetime(2023, 6, 15), model_name="graphcast")
        assert result is None

    def test_unknown_model(self):
        result = run_forecast(datetime(2023, 6, 15), model_name="nonexistent")
        assert result is None


class TestSyntheticForecast:
    def test_returns_dataset(self):
        ds = generate_synthetic_forecast(datetime(2023, 6, 15), nsteps=2)
        assert "t2m" in ds
        assert "u10m" in ds

    def test_has_time_steps(self):
        ds = generate_synthetic_forecast(datetime(2023, 6, 15), nsteps=4)
        assert ds["t2m"].shape[0] == 4

    def test_custom_variables(self):
        ds = generate_synthetic_forecast(
            datetime(2023, 6, 15), variables=["u10m", "v10m"]
        )
        assert "u10m" in ds
        assert "t2m" not in ds
