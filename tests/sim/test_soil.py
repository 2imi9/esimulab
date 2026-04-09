"""Tests for MPM soil module."""

from unittest.mock import MagicMock

from esimulab.sim.soil import (
    LANDCOVER_TO_SOIL,
    SoilConfig,
    soil_config_from_temperature,
)


class TestSoilConfig:
    def test_default_config(self):
        config = SoilConfig()
        assert config.material_type == "sand"
        assert config.thickness == 2.0

    def test_custom_config(self):
        config = SoilConfig(material_type="snow", thickness=1.0)
        assert config.material_type == "snow"


class TestSoilConfigFromTemperature:
    def test_freezing_returns_snow(self):
        config = soil_config_from_temperature(250.0)  # -23°C
        assert config.material_type == "snow"

    def test_cold_returns_elastic(self):
        config = soil_config_from_temperature(270.0)  # -3°C
        assert config.material_type == "elastic"

    def test_moderate_returns_sand(self):
        config = soil_config_from_temperature(285.0)  # 12°C
        assert config.material_type == "sand"

    def test_warm_returns_sand(self):
        config = soil_config_from_temperature(300.0)  # 27°C
        assert config.material_type == "sand"


class TestLandcoverMapping:
    def test_water_has_no_soil(self):
        assert LANDCOVER_TO_SOIL[80] is None

    def test_forest_is_elastic(self):
        assert LANDCOVER_TO_SOIL[10] == "elastic"

    def test_wetland_is_liquid(self):
        assert LANDCOVER_TO_SOIL[90] == "liquid"

    def test_snow_is_snow(self):
        assert LANDCOVER_TO_SOIL[70] == "snow"


class TestAddSoilLayer:
    def test_add_soil_creates_entity(self):
        from esimulab.sim.soil import add_soil_layer

        gs = MagicMock()
        scene = MagicMock()
        hf = MagicMock()
        hf.bounds_min = (0.0, 0.0, 0.0)
        hf.bounds_max = (100.0, 100.0, 50.0)

        add_soil_layer(gs, scene, hf, SoilConfig(material_type="sand"))

        scene.add_entity.assert_called_once()
        gs.materials.MPM.Sand.assert_called_once()
