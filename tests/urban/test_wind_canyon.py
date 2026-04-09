"""Tests for urban wind canyon effects."""

import numpy as np

from esimulab.urban.wind_canyon import (
    compute_canyon_speedup,
    compute_urban_turbulence,
    create_urban_wind_zones,
)


class TestCanyonSpeedup:
    def test_no_buildings_no_effect(self):
        heights = np.zeros((20, 20), dtype=np.float32)
        speedup = compute_canyon_speedup(heights, (1.0, 0.0), pixel_size=10.0)
        np.testing.assert_allclose(speedup, 1.0)

    def test_buildings_modify_wind(self):
        heights = np.zeros((20, 20), dtype=np.float32)
        heights[8:12, 8:12] = 30.0  # building block
        speedup = compute_canyon_speedup(heights, (1.0, 0.0), pixel_size=10.0)
        # Inside buildings should be zero
        assert speedup[10, 10] == 0.0
        # Some cells should differ from 1.0
        assert not np.allclose(speedup, 1.0)

    def test_zero_inside_buildings(self):
        heights = np.zeros((10, 10), dtype=np.float32)
        heights[4:6, 4:6] = 20.0
        speedup = compute_canyon_speedup(heights, (1.0, 0.0), pixel_size=5.0)
        assert speedup[4, 4] == 0.0
        assert speedup[5, 5] == 0.0


class TestUrbanTurbulence:
    def test_no_buildings_no_turbulence(self):
        heights = np.zeros((15, 15), dtype=np.float32)
        turb = compute_urban_turbulence(heights, wind_speed=5.0, pixel_size=10.0)
        np.testing.assert_allclose(turb, 0.0)

    def test_buildings_create_turbulence(self):
        heights = np.zeros((15, 15), dtype=np.float32)
        heights[6:9, 6:9] = 25.0
        turb = compute_urban_turbulence(heights, wind_speed=5.0, pixel_size=10.0)
        # Near-building cells should have turbulence
        assert turb.max() > 0

    def test_inside_buildings_no_turbulence(self):
        heights = np.zeros((15, 15), dtype=np.float32)
        heights[6:9, 6:9] = 25.0
        turb = compute_urban_turbulence(heights, wind_speed=5.0, pixel_size=10.0)
        assert turb[7, 7] == 0.0


class TestUrbanWindZones:
    def test_creates_zones(self):
        heights = np.zeros((30, 30), dtype=np.float32)
        heights[10:15, 10:15] = 20.0
        zones = create_urban_wind_zones(
            heights,
            base_wind_dir=(1.0, 0.0, 0.0),
            base_wind_speed=5.0,
            pixel_size=10.0,
            heightfield_bounds=((-150, -150, 0), (150, 150, 50)),
        )
        assert len(zones) == 9  # 3x3 grid
        assert all("strength" in z for z in zones)
