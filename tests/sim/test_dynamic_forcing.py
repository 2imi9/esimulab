"""Tests for dynamic atmospheric forcing."""

from unittest.mock import MagicMock

from esimulab.sim.dynamic_forcing import (
    apply_forcing_at_step,
    create_forcing_schedule,
)


class TestCreateSchedule:
    def test_creates_correct_intervals(self):
        schedule = create_forcing_schedule(
            wind_dir=(1.0, 0.0, 0.0),
            wind_mag=5.0,
            precip_rate=2.0,
            num_steps=1000,
            interval_steps=100,
        )
        assert len(schedule.wind_sequence) == 10
        assert len(schedule.precip_sequence) == 10

    def test_wind_rotates(self):
        schedule = create_forcing_schedule(
            wind_dir=(1.0, 0.0, 0.0),
            wind_mag=5.0,
            precip_rate=0.0,
            num_steps=200,
            interval_steps=100,
            wind_rotation_deg=90.0,
        )
        first_dir = schedule.wind_sequence[0][0]
        last_dir = schedule.wind_sequence[-1][0]
        # Direction should have changed
        assert abs(first_dir[0] - last_dir[0]) > 0.1 or abs(first_dir[1] - last_dir[1]) > 0.1

    def test_precip_peaks_at_middle(self):
        schedule = create_forcing_schedule(
            wind_dir=(1.0, 0.0, 0.0),
            wind_mag=3.0,
            precip_rate=5.0,
            num_steps=500,
            interval_steps=100,
        )
        rates = [p[0] for p in schedule.precip_sequence]
        # Middle values should be higher than start/end
        mid_idx = len(rates) // 2
        assert rates[mid_idx] > rates[0]


class TestApplyForcing:
    def test_emitter_called(self):
        schedule = create_forcing_schedule(
            wind_dir=(1.0, 0.0, 0.0), wind_mag=5.0,
            precip_rate=3.0, num_steps=100, interval_steps=50,
        )
        emitter = MagicMock()

        # Step 50 is at mid-sim where precip peaks (sin(pi/2) > 0)
        result = apply_forcing_at_step(schedule, step=50, emitter=emitter, z_top=200)
        emitter.emit.assert_called_once()
        assert result["precip_rate"] > 0

    def test_no_emitter_ok(self):
        schedule = create_forcing_schedule(
            wind_dir=(1.0, 0.0, 0.0), wind_mag=5.0,
            precip_rate=0.0, num_steps=100, interval_steps=50,
        )
        result = apply_forcing_at_step(schedule, step=50, emitter=None)
        assert "wind_magnitude" in result

    def test_returns_current_params(self):
        schedule = create_forcing_schedule(
            wind_dir=(0.0, 1.0, 0.0), wind_mag=10.0,
            precip_rate=1.0, num_steps=200, interval_steps=100,
        )
        r0 = apply_forcing_at_step(schedule, step=0)
        r1 = apply_forcing_at_step(schedule, step=150)
        assert r0["interval"] == 0
        assert r1["interval"] == 1
