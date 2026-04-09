"""Tests for building footprint acquisition and extrusion."""

import numpy as np

from esimulab.urban.buildings import (
    Building,
    BuildingDataset,
    extrude_buildings_to_mesh,
    fetch_building_footprints,
)


class TestBuildingDataset:
    def test_count(self):
        ds = BuildingDataset(buildings=[
            Building(footprint=np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), height=10),
        ])
        assert ds.count == 1

    def test_footprint_area(self):
        fp = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=np.float64)
        ds = BuildingDataset(buildings=[Building(footprint=fp, height=5)])
        assert abs(ds.total_footprint_area - 100.0) < 1.0


class TestFetchBuildings:
    def test_synthetic_source(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = fetch_building_footprints(bbox, source="synthetic")
        assert ds.count > 0
        assert ds.source == "synthetic"

    def test_synthetic_has_varied_heights(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = fetch_building_footprints(bbox, source="synthetic")
        heights = [b.height for b in ds.buildings]
        assert max(heights) > min(heights)  # heights vary


class TestExtrudeMesh:
    def test_extrude_single_building(self):
        fp = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=np.float64)
        ds = BuildingDataset(buildings=[Building(footprint=fp, height=20)])
        verts, faces = extrude_buildings_to_mesh(ds)
        assert verts.shape == (8, 3)  # 4 bottom + 4 top
        assert faces.shape == (12, 3)  # 6 sides × 2 triangles

    def test_extrude_multiple(self):
        buildings = [
            Building(
                footprint=np.array([[i*20, 0], [i*20+10, 0], [i*20+10, 10], [i*20, 10], [i*20, 0]],
                                   dtype=np.float64),
                height=10 + i * 5,
            )
            for i in range(3)
        ]
        ds = BuildingDataset(buildings=buildings)
        verts, faces = extrude_buildings_to_mesh(ds)
        assert verts.shape[0] == 24  # 3 buildings × 8 verts
        assert faces.shape[0] == 36  # 3 buildings × 12 faces

    def test_empty_dataset(self):
        ds = BuildingDataset()
        verts, faces = extrude_buildings_to_mesh(ds)
        assert verts.shape[0] == 0
