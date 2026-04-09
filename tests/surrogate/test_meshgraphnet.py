"""Tests for MeshGraphNet terrain surrogate."""

import numpy as np

from esimulab.surrogate.meshgraphnet import MGNConfig, terrain_to_graph


class TestMGNConfig:
    def test_defaults(self):
        config = MGNConfig()
        assert config.input_dim_nodes == 6
        assert config.output_dim == 3
        assert config.processor_size == 15


class TestTerrainToGraph:
    def test_basic_graph(self):
        dem = np.random.rand(20, 20).astype(np.float32) * 100
        graph = terrain_to_graph(dem, pixel_size=30.0)

        assert "node_features" in graph
        assert "edge_index" in graph
        assert "edge_features" in graph

    def test_node_count(self):
        dem = np.random.rand(10, 10).astype(np.float32)
        graph = terrain_to_graph(dem, pixel_size=10.0)

        assert graph["node_features"].shape[0] == 100  # 10x10
        assert graph["node_features"].shape[1] == 6

    def test_edge_features_shape(self):
        dem = np.random.rand(8, 8).astype(np.float32)
        graph = terrain_to_graph(dem, pixel_size=5.0)

        assert graph["edge_features"].shape[1] == 3  # dx, dy, dist

    def test_subsample_large_terrain(self):
        dem = np.random.rand(500, 500).astype(np.float32)
        graph = terrain_to_graph(dem, pixel_size=30.0, max_nodes=10000)

        assert graph["node_features"].shape[0] <= 10000

    def test_edge_distances_positive(self):
        dem = np.zeros((5, 5), dtype=np.float32)
        graph = terrain_to_graph(dem, pixel_size=10.0)

        distances = graph["edge_features"][:, 2]
        assert np.all(distances > 0)
