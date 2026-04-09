"""Tests for DEM-to-mesh conversion."""

import numpy as np

from esimulab.terrain.mesh import (
    dem_to_mesh,
    dem_to_mesh_vectorized,
    export_obj,
)


class TestDemToMesh:
    def test_basic_mesh(self):
        dem = np.ones((5, 5), dtype=np.float32) * 100
        verts, faces = dem_to_mesh(dem, pixel_size=10.0)
        assert verts.shape == (25, 3)
        assert faces.shape[0] == 32  # (5-1)*(5-1)*2

    def test_vertex_positions(self):
        dem = np.array([[0, 0], [0, 10]], dtype=np.float32)
        verts, _ = dem_to_mesh(dem, pixel_size=5.0)
        # Check z values match DEM
        assert verts[3, 2] == 10.0  # bottom-right

    def test_origin_offset(self):
        dem = np.zeros((3, 3), dtype=np.float32)
        verts, _ = dem_to_mesh(dem, pixel_size=1.0, origin=(100.0, 200.0))
        assert verts[0, 0] == 100.0
        assert verts[0, 1] == 200.0


class TestVectorizedMesh:
    def test_matches_loop_version(self):
        dem = np.random.rand(10, 10).astype(np.float32) * 50
        v1, f1 = dem_to_mesh(dem, pixel_size=5.0)
        v2, f2 = dem_to_mesh_vectorized(dem, pixel_size=5.0)
        np.testing.assert_array_equal(v1, v2)
        assert f1.shape == f2.shape

    def test_large_terrain(self):
        dem = np.random.rand(100, 100).astype(np.float32)
        verts, faces = dem_to_mesh_vectorized(dem, pixel_size=30.0)
        assert verts.shape[0] == 10000
        assert faces.shape[0] == 99 * 99 * 2


class TestExportObj:
    def test_creates_file(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        path = export_obj(verts, faces, tmp_path / "test.obj")
        assert path.exists()
        content = path.read_text()
        assert "v " in content
        assert "f " in content
