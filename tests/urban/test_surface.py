"""Tests for impervious surface and urban hydrology."""

import numpy as np

from esimulab.urban.surface import (
    compute_impervious_fraction,
    urban_heat_island_adjustment,
    urban_infiltration_rate,
    urban_runoff_coefficient,
)


class TestImperviousFraction:
    def test_forest_low_imperviousness(self):
        lc = np.full((10, 10), 10, dtype=np.uint8)  # tree cover
        result = compute_impervious_fraction(lc)
        assert result.mean() < 0.1

    def test_built_up_high_imperviousness(self):
        lc = np.full((10, 10), 50, dtype=np.uint8)  # built-up
        result = compute_impervious_fraction(lc)
        assert result.mean() > 0.5

    def test_building_mask_overrides(self):
        lc = np.full((10, 10), 30, dtype=np.uint8)  # grassland
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1  # buildings in center

        result = compute_impervious_fraction(lc, building_mask=mask)
        assert result[5, 5] > 0.9  # building area
        assert result[0, 0] < 0.1  # open area


class TestRunoffCoefficient:
    def test_urban_higher_than_rural(self):
        urban = np.full((10, 10), 50, dtype=np.uint8)
        rural = np.full((10, 10), 10, dtype=np.uint8)
        c_urban = urban_runoff_coefficient(urban).mean()
        c_rural = urban_runoff_coefficient(rural).mean()
        assert c_urban > c_rural

    def test_range_0_to_1(self):
        lc = np.random.choice([10, 30, 50, 80], size=(20, 20)).astype(np.uint8)
        result = urban_runoff_coefficient(lc)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestInfiltration:
    def test_forest_high_infiltration(self):
        lc = np.full((5, 5), 10, dtype=np.uint8)
        result = urban_infiltration_rate(lc)
        assert result.mean() > 20  # mm/hr

    def test_built_up_low_infiltration(self):
        lc = np.full((5, 5), 50, dtype=np.uint8)
        result = urban_infiltration_rate(lc)
        assert result.mean() < 5


class TestUrbanHeatIsland:
    def test_no_uhi_in_rural(self):
        imp = np.zeros((10, 10), dtype=np.float32)
        adj = urban_heat_island_adjustment(288.0, imp)
        np.testing.assert_allclose(adj, 0.0, atol=0.1)

    def test_positive_uhi_in_urban(self):
        imp = np.ones((10, 10), dtype=np.float32) * 0.8
        adj = urban_heat_island_adjustment(288.0, imp)
        assert adj.mean() > 0

    def test_nighttime_stronger(self):
        imp = np.full((10, 10), 0.7, dtype=np.float32)
        night = urban_heat_island_adjustment(288.0, imp, time_of_day_hour=2)
        day = urban_heat_island_adjustment(288.0, imp, time_of_day_hour=14)
        assert night.mean() > day.mean()
