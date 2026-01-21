"""Tests for task definitions and queue management."""

import numpy as np
import pytest

from satellite_edge.environment.tasks import (
    Task, TaskType, TaskQueue, TaskSpec, TASK_SPECS
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types_exist(self):
        assert TaskType.DETECTION == 0
        assert TaskType.ANOMALY == 1
        assert TaskType.CLOUD_MASK == 2
        assert TaskType.COMPRESSION == 3

    def test_all_types_have_specs(self):
        for task_type in TaskType:
            assert task_type in TASK_SPECS
            spec = TASK_SPECS[task_type]
            assert isinstance(spec, TaskSpec)
            assert spec.compute_cost > 0
            assert spec.memory_footprint > 0
            assert spec.base_value > 0


class TestTaskSpec:
    """Tests for task specifications."""

    def test_spec_values_are_reasonable(self):
        for task_type, spec in TASK_SPECS.items():
            # Compute costs should be in realistic TOPS range
            assert 0.1 <= spec.compute_cost <= 20.0

            # Memory footprint in GB, reasonable for satellite
            assert 0.01 <= spec.memory_footprint <= 2.0

            # Value should be positive
            assert spec.base_value > 0

            # Deadline sensitivity in [0, 1]
            assert 0 <= spec.deadline_sensitivity <= 1.0


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        task = Task(
            task_type=TaskType.DETECTION,
            tile_id=42,
            arrival_time=100.0,
        )
        assert task.task_type == TaskType.DETECTION
        assert task.tile_id == 42
        assert task.arrival_time == 100.0
        assert task.priority_boost == 1.0
        assert task.progress == 0.0

    def test_task_spec_property(self):
        task = Task(TaskType.ANOMALY, tile_id=1, arrival_time=0.0)
        spec = task.spec
        assert spec == TASK_SPECS[TaskType.ANOMALY]

    def test_remaining_compute(self):
        task = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        initial_remaining = task.remaining_compute
        assert initial_remaining == task.spec.compute_cost

        task.progress = 0.5
        assert task.remaining_compute == pytest.approx(initial_remaining * 0.5)

        task.progress = 1.0
        assert task.remaining_compute == pytest.approx(0.0)

    def test_current_value_no_decay(self):
        task = Task(TaskType.CLOUD_MASK, tile_id=1, arrival_time=0.0)
        # Cloud mask has low deadline sensitivity (0.3)
        value_t0 = task.current_value(0.0)
        value_t10 = task.current_value(10.0)

        # Should have some decay but not much
        assert value_t10 < value_t0
        assert value_t10 > value_t0 * 0.5  # Not too much decay

    def test_current_value_high_decay(self):
        task = Task(TaskType.ANOMALY, tile_id=1, arrival_time=0.0)
        # Anomaly has high deadline sensitivity (0.9)
        value_t0 = task.current_value(0.0)
        value_t20 = task.current_value(20.0)

        # Should decay significantly
        assert value_t20 < value_t0 * 0.5

    def test_current_value_with_priority_boost(self):
        task_normal = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        task_boosted = Task(
            TaskType.DETECTION, tile_id=2, arrival_time=0.0, priority_boost=3.0
        )

        value_normal = task_normal.current_value(0.0)
        value_boosted = task_boosted.current_value(0.0)

        assert value_boosted == pytest.approx(value_normal * 3.0)

    def test_apply_compute_partial(self):
        task = Task(TaskType.COMPRESSION, tile_id=1, arrival_time=0.0)
        # Compression costs 4.0 TOPS

        tops_used = task.apply_compute(tops_available=2.0, timestep_duration=1.0)

        assert tops_used == 2.0
        assert task.progress == pytest.approx(0.5)
        assert not task.is_complete

    def test_apply_compute_completes_task(self):
        task = Task(TaskType.CLOUD_MASK, tile_id=1, arrival_time=0.0)
        # Cloud mask costs 1.5 TOPS

        tops_used = task.apply_compute(tops_available=10.0, timestep_duration=1.0)

        assert tops_used == pytest.approx(1.5)
        assert task.progress == pytest.approx(1.0)
        assert task.is_complete

    def test_is_complete(self):
        task = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        assert not task.is_complete

        task.progress = 0.99999
        assert not task.is_complete

        task.progress = 1.0
        assert task.is_complete

    def test_unique_ids(self):
        task1 = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        task2 = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        assert task1._id != task2._id


class TestTaskQueue:
    """Tests for TaskQueue."""

    def test_empty_queue(self):
        queue = TaskQueue(max_size=10)
        assert len(queue) == 0
        assert queue.is_empty
        assert not queue.is_full

    def test_add_task(self):
        queue = TaskQueue(max_size=10)
        task = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)

        result = queue.add(task)

        assert result is True
        assert len(queue) == 1
        assert not queue.is_empty

    def test_add_to_full_queue(self):
        queue = TaskQueue(max_size=2)
        task1 = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        task2 = Task(TaskType.ANOMALY, tile_id=2, arrival_time=0.0)
        task3 = Task(TaskType.COMPRESSION, tile_id=3, arrival_time=0.0)

        queue.add(task1)
        queue.add(task2)
        result = queue.add(task3)

        assert result is False
        assert queue.is_full
        assert len(queue) == 2

    def test_remove_task(self):
        queue = TaskQueue()
        task = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        queue.add(task)

        queue.remove(task)

        assert len(queue) == 0

    def test_remove_completed(self):
        queue = TaskQueue()
        task1 = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        task2 = Task(TaskType.ANOMALY, tile_id=2, arrival_time=0.0)
        task2.progress = 1.0  # Mark as complete

        queue.add(task1)
        queue.add(task2)

        completed = queue.remove_completed()

        assert len(completed) == 1
        assert completed[0] == task2
        assert len(queue) == 1

    def test_get_by_type(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.DETECTION, tile_id=2, arrival_time=0.0))
        queue.add(Task(TaskType.ANOMALY, tile_id=3, arrival_time=0.0))

        detection_tasks = queue.get_by_type(TaskType.DETECTION)
        anomaly_tasks = queue.get_by_type(TaskType.ANOMALY)

        assert len(detection_tasks) == 2
        assert len(anomaly_tasks) == 1

    def test_get_task_at_index(self):
        queue = TaskQueue()
        task = Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0)
        queue.add(task)

        assert queue.get_task_at_index(0) == task
        assert queue.get_task_at_index(1) is None
        assert queue.get_task_at_index(-1) is None

    def test_get_queue_depths(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.DETECTION, tile_id=2, arrival_time=0.0))
        queue.add(Task(TaskType.ANOMALY, tile_id=3, arrival_time=0.0))
        queue.add(Task(TaskType.COMPRESSION, tile_id=4, arrival_time=0.0))

        depths = queue.get_queue_depths()

        assert depths[TaskType.DETECTION] == 2
        assert depths[TaskType.ANOMALY] == 1
        assert depths[TaskType.CLOUD_MASK] == 0
        assert depths[TaskType.COMPRESSION] == 1

    def test_get_total_compute_by_type(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.DETECTION, tile_id=2, arrival_time=0.0))

        compute = queue.get_total_compute_by_type()

        expected_detection = TASK_SPECS[TaskType.DETECTION].compute_cost * 2
        assert compute[TaskType.DETECTION] == pytest.approx(expected_detection)
        assert compute[TaskType.ANOMALY] == 0.0

    def test_get_total_memory(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.ANOMALY, tile_id=2, arrival_time=0.0))

        total_memory = queue.get_total_memory()

        expected = (
            TASK_SPECS[TaskType.DETECTION].memory_footprint +
            TASK_SPECS[TaskType.ANOMALY].memory_footprint
        )
        assert total_memory == pytest.approx(expected)

    def test_iteration(self):
        queue = TaskQueue()
        tasks = [
            Task(TaskType.DETECTION, tile_id=i, arrival_time=float(i))
            for i in range(3)
        ]
        for task in tasks:
            queue.add(task)

        iterated = list(queue)
        assert iterated == tasks

    def test_clear(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.ANOMALY, tile_id=2, arrival_time=0.0))

        queue.clear()

        assert len(queue) == 0
        assert queue.is_empty

    def test_get_oldest_by_type(self):
        queue = TaskQueue()
        queue.add(Task(TaskType.DETECTION, tile_id=1, arrival_time=0.0))
        queue.add(Task(TaskType.DETECTION, tile_id=2, arrival_time=5.0))
        queue.add(Task(TaskType.ANOMALY, tile_id=3, arrival_time=3.0))

        oldest = queue.get_oldest_by_type()

        # Oldest detection task arrived at 5.0, oldest anomaly at 3.0
        assert oldest[TaskType.DETECTION] == 5.0
        assert oldest[TaskType.ANOMALY] == 3.0
        assert oldest[TaskType.CLOUD_MASK] == 0.0  # No tasks
        assert oldest[TaskType.COMPRESSION] == 0.0

    def test_get_oldest_by_type_empty_queue(self):
        queue = TaskQueue()
        oldest = queue.get_oldest_by_type()

        # All zeros when empty
        for task_type in TaskType:
            assert oldest[task_type] == 0.0
