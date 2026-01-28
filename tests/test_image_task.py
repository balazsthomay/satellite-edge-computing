"""Tests for ImageTask extension."""

import numpy as np
import pytest

from satellite_edge.environment import ImageTask, Task, TaskType


class TestImageTask:
    """Tests for ImageTask class."""

    def test_inherits_from_task(self):
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
        )
        assert isinstance(task, Task)

    def test_default_image_metrics(self):
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
        )
        assert task.cloud_fraction == 0.0
        assert task.entropy_score == 0.5
        assert task.edge_density == 0.5
        assert task.tile_path is None

    def test_custom_image_metrics(self):
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.3,
            entropy_score=0.7,
            edge_density=0.8,
            tile_path="/path/to/tile.tif",
        )
        assert task.cloud_fraction == 0.3
        assert task.entropy_score == 0.7
        assert task.edge_density == 0.8
        assert task.tile_path == "/path/to/tile.tif"

    def test_quality_multiplier_detection(self):
        # Clear image with edges = high value for detection
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
            entropy_score=0.5,
            edge_density=1.0,
        )
        mult = task.image_quality_multiplier
        # (1-0) * (0.3*0.5 + 0.7*1.0) = 0.85
        assert abs(mult - 0.85) < 0.01

        # Cloudy image = low value for detection
        task_cloudy = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.9,
            entropy_score=0.5,
            edge_density=1.0,
        )
        mult_cloudy = task_cloudy.image_quality_multiplier
        # (1-0.9) * (0.3*0.5 + 0.7*1.0) = 0.085 -> floored to 0.1
        assert mult_cloudy == 0.1

    def test_quality_multiplier_anomaly(self):
        task = ImageTask(
            task_type=TaskType.ANOMALY,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
            entropy_score=0.8,
            edge_density=0.6,
        )
        mult = task.image_quality_multiplier
        # (1-0) * (0.5*0.8 + 0.5*0.6) = 0.7
        assert abs(mult - 0.7) < 0.01

    def test_quality_multiplier_cloud_mask(self):
        # More clouds = more value for cloud masking
        clear_task = ImageTask(
            task_type=TaskType.CLOUD_MASK,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
        )
        cloudy_task = ImageTask(
            task_type=TaskType.CLOUD_MASK,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=1.0,
        )

        clear_mult = clear_task.image_quality_multiplier
        cloudy_mult = cloudy_task.image_quality_multiplier

        # Clear: 0.2 + 0.8*0 = 0.2
        assert abs(clear_mult - 0.2) < 0.01
        # Cloudy: 0.2 + 0.8*1.0 = 1.0
        assert abs(cloudy_mult - 1.0) < 0.01

    def test_quality_multiplier_compression(self):
        # Higher entropy = better compression candidate
        low_entropy = ImageTask(
            task_type=TaskType.COMPRESSION,
            tile_id=0,
            arrival_time=0.0,
            entropy_score=0.0,
        )
        high_entropy = ImageTask(
            task_type=TaskType.COMPRESSION,
            tile_id=0,
            arrival_time=0.0,
            entropy_score=1.0,
        )

        low_mult = low_entropy.image_quality_multiplier
        high_mult = high_entropy.image_quality_multiplier

        # Low: 0.3 + 0.7*0 = 0.3
        assert abs(low_mult - 0.3) < 0.01
        # High: 0.3 + 0.7*1.0 = 1.0
        assert abs(high_mult - 1.0) < 0.01

    def test_quality_multiplier_floor(self):
        # Ensure multiplier doesn't go below 0.1
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=1.0,  # 100% clouds
            entropy_score=0.0,
            edge_density=0.0,
        )
        mult = task.image_quality_multiplier
        assert mult == 0.1

    def test_current_value_includes_quality(self):
        # Base task value
        base_task = Task(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
        )
        base_value = base_task.current_value(0.0)

        # ImageTask with quality multiplier
        image_task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
            entropy_score=0.5,
            edge_density=1.0,
        )
        image_value = image_task.current_value(0.0)
        expected_mult = image_task.image_quality_multiplier

        # ImageTask value = base value * quality multiplier
        assert abs(image_value - base_value * expected_mult) < 0.01

    def test_current_value_with_decay(self):
        task = ImageTask(
            task_type=TaskType.ANOMALY,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
            entropy_score=0.8,
            edge_density=0.8,
        )

        value_at_0 = task.current_value(0.0)
        value_at_10 = task.current_value(10.0)

        # Value should decay over time
        assert value_at_10 < value_at_0

    def test_current_value_with_priority_boost(self):
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            priority_boost=2.0,
            cloud_fraction=0.0,
            entropy_score=0.5,
            edge_density=0.5,
        )

        # Priority boost should be applied
        value = task.current_value(0.0)
        assert value > 0

    def test_inherits_task_properties(self):
        task = ImageTask(
            task_type=TaskType.COMPRESSION,
            tile_id=42,
            arrival_time=5.0,
            priority_boost=1.5,
        )

        # Should have all Task properties
        assert task.spec is not None
        assert task.remaining_compute > 0
        assert not task.is_complete

    def test_apply_compute(self):
        task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
        )

        # Apply some compute
        initial_remaining = task.remaining_compute
        task.apply_compute(3.0, 10.0)

        assert task.remaining_compute < initial_remaining
        assert task.progress > 0


class TestImageTaskComparison:
    """Test ImageTask behavior compared to regular Task."""

    def test_same_base_specs(self):
        task = Task(task_type=TaskType.DETECTION, tile_id=0, arrival_time=0.0)
        image_task = ImageTask(
            task_type=TaskType.DETECTION, tile_id=0, arrival_time=0.0
        )

        assert task.spec == image_task.spec

    def test_value_difference_from_quality(self):
        # Task with default quality (neutral)
        default_task = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            # defaults: cloud=0, entropy=0.5, edges=0.5
        )

        # High quality task
        high_quality = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.0,
            entropy_score=1.0,
            edge_density=1.0,
        )

        # Low quality task (cloudy)
        low_quality = ImageTask(
            task_type=TaskType.DETECTION,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=0.9,
            entropy_score=0.1,
            edge_density=0.1,
        )

        default_value = default_task.current_value(0.0)
        high_value = high_quality.current_value(0.0)
        low_value = low_quality.current_value(0.0)

        assert high_value > default_value
        assert default_value > low_value
