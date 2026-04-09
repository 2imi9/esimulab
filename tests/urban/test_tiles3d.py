"""Tests for Google 3D Tiles integration."""

import numpy as np

from esimulab.urban.tiles3d import (
    CityMesh,
    Tile3DConfig,
    _bv_intersects_bbox,
    fetch_3d_tiles,
    generate_synthetic_city_mesh,
)


class TestTile3DConfig:
    def test_defaults(self):
        config = Tile3DConfig()
        assert config.max_tiles == 50
        assert config.api_key == ""


class TestBvIntersection:
    def test_region_intersection(self):
        # Region in radians: LA area
        import math

        bv = {"region": [
            math.radians(-118.3), math.radians(34.0),
            math.radians(-118.2), math.radians(34.1),
            0, 500,
        ]}
        bbox = (-118.25, 34.05, -118.15, 34.15)
        assert _bv_intersects_bbox(bv, bbox) is True

    def test_no_intersection(self):
        import math

        bv = {"region": [
            math.radians(0), math.radians(0),
            math.radians(1), math.radians(1),
            0, 100,
        ]}
        bbox = (-118.3, 34.0, -118.2, 34.1)
        assert _bv_intersects_bbox(bv, bbox) is False

    def test_box_bv_conservative(self):
        # Box or sphere: always returns True (conservative)
        bv = {"box": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]}
        assert _bv_intersects_bbox(bv, (-118, 34, -117, 35)) is True


class TestSyntheticCity:
    def test_generates_mesh(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        mesh = generate_synthetic_city_mesh(bbox, building_count=10)
        assert isinstance(mesh, CityMesh)
        assert mesh.vertices.shape[0] == 80  # 10 buildings * 8 verts
        assert mesh.faces.shape[0] == 120  # 10 buildings * 12 faces
        assert mesh.source == "synthetic"

    def test_building_heights_vary(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        mesh = generate_synthetic_city_mesh(bbox, building_count=50)
        # Z values should vary (different building heights)
        z_vals = mesh.vertices[:, 2]
        assert z_vals.max() > 10  # some tall buildings

    def test_deterministic(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        m1 = generate_synthetic_city_mesh(bbox, seed=42)
        m2 = generate_synthetic_city_mesh(bbox, seed=42)
        np.testing.assert_array_equal(m1.vertices, m2.vertices)


class TestFetch3DTiles:
    def test_no_api_key_returns_synthetic(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        mesh = fetch_3d_tiles(bbox, Tile3DConfig(api_key=""))
        assert mesh is not None
        assert mesh.source == "synthetic"

    def test_city_mesh_properties(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        mesh = fetch_3d_tiles(bbox)
        assert mesh.bounds == bbox
        assert mesh.total_triangles > 0
        assert mesh.vertices.dtype == np.float32
