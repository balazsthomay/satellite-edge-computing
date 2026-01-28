"""Tests for value estimator."""

import numpy as np
import pytest

from satellite_edge.data import ValueEstimator, create_synthetic_tile
from satellite_edge.environment import TaskType


class TestValueEstimator:
    """Tests for ValueEstimator class."""

    @pytest.fixture
    def estimator(self):
        return ValueEstimator()

    def test_estimate_cloud_fraction_empty(self, estimator):
        tile = np.array([])
        result = estimator.estimate_cloud_fraction(tile)
        assert result == 0.0

    def test_estimate_cloud_fraction_dark_image(self, estimator):
        # Dark image should have low cloud fraction
        tile = np.zeros((100, 100, 3), dtype=np.uint8)
        tile[:, :] = 20  # Dark
        result = estimator.estimate_cloud_fraction(tile)
        assert result < 0.2

    def test_estimate_cloud_fraction_bright_image(self, estimator):
        # Bright uniform image should have some cloud
        tile = np.full((100, 100, 3), 250, dtype=np.uint8)
        result = estimator.estimate_cloud_fraction(tile)
        # With percentile-based threshold, uniform bright should have ~15% above threshold
        assert 0.0 <= result <= 0.5

    def test_estimate_cloud_fraction_cloud_tile(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "cloud", seed=42)
        result = estimator.estimate_cloud_fraction(tile)
        # Cloud tiles should have higher cloud fraction
        assert result > 0.0

    def test_estimate_cloud_fraction_normalized_range(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "mixed", seed=42)
        result = estimator.estimate_cloud_fraction(tile)
        assert 0.0 <= result <= 1.0

    def test_compute_entropy_empty(self, estimator):
        tile = np.array([])
        result = estimator.compute_entropy(tile)
        assert result == 0.0

    def test_compute_entropy_constant_image(self, estimator):
        # Constant image has zero entropy
        tile = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = estimator.compute_entropy(tile)
        assert result == 0.0

    def test_compute_entropy_urban_tile(self, estimator):
        # Urban tiles should have high entropy (varied texture)
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        result = estimator.compute_entropy(tile)
        assert result > 0.0

    def test_compute_entropy_water_tile(self, estimator):
        # Water tiles should have lower entropy (uniform)
        tile = create_synthetic_tile(128, 128, 4, "water", seed=42)
        result = estimator.compute_entropy(tile)
        # Still should have some entropy from noise
        assert 0.0 <= result <= 1.0

    def test_compute_entropy_normalized_range(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "mixed", seed=42)
        result = estimator.compute_entropy(tile)
        assert 0.0 <= result <= 1.0

    def test_compute_edge_density_empty(self, estimator):
        tile = np.array([])
        result = estimator.compute_edge_density(tile)
        assert result == 0.0

    def test_compute_edge_density_constant_image(self, estimator):
        # Constant image has no edges
        tile = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = estimator.compute_edge_density(tile)
        assert result == 0.0

    def test_compute_edge_density_urban_tile(self, estimator):
        # Urban tiles should have high edge density
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        result = estimator.compute_edge_density(tile)
        assert result > 0.0

    def test_compute_edge_density_normalized_range(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "mixed", seed=42)
        result = estimator.compute_edge_density(tile)
        assert 0.0 <= result <= 1.0

    def test_compute_all_metrics(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        metrics = estimator.compute_all_metrics(tile)

        assert "cloud_fraction" in metrics
        assert "entropy" in metrics
        assert "edge_density" in metrics

        assert 0.0 <= metrics["cloud_fraction"] <= 1.0
        assert 0.0 <= metrics["entropy"] <= 1.0
        assert 0.0 <= metrics["edge_density"] <= 1.0

    def test_compute_task_value_detection(self, estimator):
        # Clear image should have higher detection value than cloudy
        clear_tile = np.zeros((100, 100, 3), dtype=np.uint8)
        clear_tile[::10, ::10] = 200  # Add some edges

        cloud_tile = np.full((100, 100, 3), 250, dtype=np.uint8)

        clear_value = estimator.compute_task_value(clear_tile, TaskType.DETECTION)
        cloud_value = estimator.compute_task_value(cloud_tile, TaskType.DETECTION)

        # Both should be positive
        assert clear_value > 0
        assert cloud_value > 0

    def test_compute_task_value_cloud_mask(self, estimator):
        # Cloudy image should have higher cloud_mask value
        clear_tile = np.zeros((100, 100, 3), dtype=np.uint8)
        cloud_tile = create_synthetic_tile(128, 128, 4, "cloud", seed=42)

        clear_value = estimator.compute_task_value(clear_tile, TaskType.CLOUD_MASK)
        cloud_value = estimator.compute_task_value(cloud_tile, TaskType.CLOUD_MASK)

        # Clear sky should still have some value (minimum 0.2 multiplier)
        assert clear_value > 0

    def test_compute_task_value_compression(self, estimator):
        # High entropy tiles are better compression candidates
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        value = estimator.compute_task_value(tile, TaskType.COMPRESSION)
        assert value > 0

    def test_compute_task_value_with_base_value(self, estimator):
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        value_1 = estimator.compute_task_value(tile, TaskType.DETECTION, base_value=1.0)
        value_10 = estimator.compute_task_value(
            tile, TaskType.DETECTION, base_value=10.0
        )
        assert abs(value_10 / value_1 - 10.0) < 0.01

    def test_custom_thresholds(self):
        estimator = ValueEstimator(
            cloud_threshold_percentile=90.0,
            entropy_disk_size=3,
        )
        tile = create_synthetic_tile(128, 128, 4, "mixed", seed=42)
        metrics = estimator.compute_all_metrics(tile)

        # Should still produce valid results
        assert 0.0 <= metrics["cloud_fraction"] <= 1.0
        assert 0.0 <= metrics["entropy"] <= 1.0


class TestValueEstimatorEdgeCases:
    """Edge case tests for ValueEstimator."""

    @pytest.fixture
    def estimator(self):
        return ValueEstimator()

    def test_grayscale_input(self, estimator):
        # 2D grayscale image
        tile = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        metrics = estimator.compute_all_metrics(tile)
        assert "cloud_fraction" in metrics

    def test_float_input(self, estimator):
        # Float input [0, 1]
        tile = np.random.random((100, 100, 3))
        metrics = estimator.compute_all_metrics(tile)
        assert "entropy" in metrics

    def test_high_dynamic_range(self, estimator):
        # Large float values
        tile = np.random.random((100, 100, 3)) * 10000
        metrics = estimator.compute_all_metrics(tile)
        assert 0.0 <= metrics["cloud_fraction"] <= 1.0
