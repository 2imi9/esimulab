"""Tests for Overture Maps building data integration."""

import numpy as np

from esimulab.urban.overture import (
    OvertureBuildingDataset,
    _generate_synthetic_overture,
    fetch_overture_buildings,
    overture_to_heightfield_mask,
)


class TestSyntheticOverture:
    def test_generates_buildings(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = _generate_synthetic_overture(bbox, max_buildings=100)
        assert ds.count == 100
        assert all(b.height > 0 for b in ds.buildings)

    def test_centroids_within_bbox(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = _generate_synthetic_overture(bbox, max_buildings=50)
        for b in ds.buildings:
            assert bbox[0] <= b.centroid[0] <= bbox[2]
            assert bbox[1] <= b.centroid[1] <= bbox[3]

    def test_heights_array(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = _generate_synthetic_overture(bbox, max_buildings=20)
        heights = ds.heights_array()
        assert heights.shape == (20,)
        assert heights.dtype == np.float32
        assert heights.min() > 0

    def test_centroids_array(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = _generate_synthetic_overture(bbox, max_buildings=10)
        centroids = ds.centroids_array()
        assert centroids.shape == (10, 2)


class TestFetchOverture:
    def test_fallback_to_synthetic(self):
        """Without network/DuckDB spatial extension, falls back to synthetic."""
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = fetch_overture_buildings(bbox, max_buildings=50)
        assert ds.count > 0
        assert ds.bbox == bbox


class TestHeightfieldMask:
    def test_creates_mask(self):
        bbox = (-118.3, 34.0, -118.2, 34.1)
        ds = _generate_synthetic_overture(bbox, max_buildings=50)
        # Use bbox as dem_bounds so coordinate mapping works
        mask = overture_to_heightfield_mask(
            ds,
            dem_shape=(100, 100),
            dem_bounds=bbox,
        )
        assert mask.shape == (100, 100)
        assert mask.max() > 0  # some buildings placed

    def test_empty_dataset(self):
        ds = OvertureBuildingDataset()
        mask = overture_to_heightfield_mask(
            ds, dem_shape=(50, 50), dem_bounds=(0, 0, 100, 100)
        )
        assert mask.sum() == 0
