"""Tests for atmospheric condition to material property mapping."""

from esimulab.atmo.material_mapping import (
    EnvironmentalMaterials,
    WaterProperties,
    materials_from_atmosphere,
    water_properties_from_temperature,
)


class TestWaterProperties:
    def test_cold_water_more_viscous(self):
        cold = water_properties_from_temperature(275.0)  # 2°C
        warm = water_properties_from_temperature(300.0)  # 27°C
        assert cold.mu > warm.mu

    def test_density_near_1000(self):
        props = water_properties_from_temperature(288.0)  # 15°C
        assert 990 < props.rho < 1005

    def test_surface_tension_positive(self):
        props = water_properties_from_temperature(293.0)  # 20°C
        assert props.gamma > 0

    def test_returns_water_properties(self):
        props = water_properties_from_temperature(288.0)
        assert isinstance(props, WaterProperties)
        assert props.stiffness > 0


class TestMaterialsFromAtmosphere:
    def test_frozen_conditions(self):
        mats = materials_from_atmosphere(250.0)  # -23°C
        assert mats.soil_type == "snow"
        assert isinstance(mats, EnvironmentalMaterials)

    def test_cold_conditions(self):
        mats = materials_from_atmosphere(268.0)  # -5°C
        assert mats.soil_type == "elastic"

    def test_saturated_conditions(self):
        mats = materials_from_atmosphere(293.0, humidity_kgm2=50.0, precipitation_mm_hr=15.0)
        assert mats.soil_type == "liquid"

    def test_hot_dry_conditions(self):
        mats = materials_from_atmosphere(308.0, humidity_kgm2=10.0)  # 35°C, dry
        assert mats.soil_type == "sand"

    def test_moderate_conditions(self):
        mats = materials_from_atmosphere(288.0, humidity_kgm2=25.0)  # 15°C
        assert mats.soil_type == "sand"

    def test_description_populated(self):
        mats = materials_from_atmosphere(288.0)
        assert len(mats.description) > 0
        assert "°C" in mats.description
