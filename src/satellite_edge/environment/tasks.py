"""Task definitions for satellite onboard CV workloads."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator
import numpy as np


class TaskType(IntEnum):
    """CV task types for onboard satellite processing.

    Each task type has different compute requirements and value profiles.
    Ordered roughly by typical priority in disaster response scenarios.
    """

    DETECTION = 0  # Object detection (ships, vehicles, aircraft)
    ANOMALY = 1  # Anomaly detection (fires, floods, changes)
    CLOUD_MASK = 2  # Cloud masking for downstream tasks
    COMPRESSION = 3  # Intelligent compression before downlink


@dataclass
class TaskSpec:
    """Specification for a task type's resource requirements."""

    compute_cost: float  # TOPS required to complete
    memory_footprint: float  # GB of buffer space needed
    base_value: float  # Nominal value when completed
    deadline_sensitivity: float  # How fast value decays (0=none, 1=linear)


# Grounded in Jiguang 1000 specs (~32 TOPS) and typical CV model requirements
# Detection: YOLOv8-nano ~6.5 TOPS for real-time
# Anomaly: Lightweight autoencoder ~2-4 TOPS
# Cloud mask: Simple U-Net ~1-2 TOPS
# Compression: Neural codec ~3-5 TOPS
TASK_SPECS: dict[TaskType, TaskSpec] = {
    TaskType.DETECTION: TaskSpec(
        compute_cost=6.0,
        memory_footprint=0.5,
        base_value=10.0,
        deadline_sensitivity=0.8,
    ),
    TaskType.ANOMALY: TaskSpec(
        compute_cost=3.0,
        memory_footprint=0.3,
        base_value=15.0,  # High value - disaster response
        deadline_sensitivity=0.9,
    ),
    TaskType.CLOUD_MASK: TaskSpec(
        compute_cost=1.5,
        memory_footprint=0.2,
        base_value=3.0,  # Enabling task, lower direct value
        deadline_sensitivity=0.3,
    ),
    TaskType.COMPRESSION: TaskSpec(
        compute_cost=4.0,
        memory_footprint=0.4,
        base_value=5.0,
        deadline_sensitivity=0.5,
    ),
}


@dataclass
class Task:
    """A single processing task in the satellite's queue."""

    task_type: TaskType
    tile_id: int  # Which image tile this task processes
    arrival_time: float  # Simulation time when task arrived
    priority_boost: float = 1.0  # Multiplier for special events (e.g., disaster)
    progress: float = 0.0  # Fraction of compute completed [0, 1]

    _id: int = field(default_factory=lambda: Task._next_id())
    _id_counter: int = field(default=0, init=False, repr=False)

    @staticmethod
    def _next_id() -> int:
        Task._id_counter = getattr(Task, "_id_counter", 0) + 1
        return Task._id_counter

    @property
    def spec(self) -> TaskSpec:
        return TASK_SPECS[self.task_type]

    @property
    def remaining_compute(self) -> float:
        """TOPS still needed to complete this task."""
        return self.spec.compute_cost * (1.0 - self.progress)

    def current_value(self, current_time: float) -> float:
        """Calculate task value with time decay.

        Value decays based on deadline_sensitivity and time since arrival.
        Uses exponential decay scaled by sensitivity parameter.
        """
        age = current_time - self.arrival_time
        decay = np.exp(-self.spec.deadline_sensitivity * age / 10.0)
        return self.spec.base_value * self.priority_boost * decay

    def apply_compute(self, tops_available: float, timestep_duration: float) -> float:
        """Apply compute to this task, return TOPS actually used."""
        tops_needed = self.remaining_compute / timestep_duration
        tops_used = min(tops_available, tops_needed)

        progress_delta = (tops_used * timestep_duration) / self.spec.compute_cost
        self.progress = min(1.0, self.progress + progress_delta)

        return tops_used

    @property
    def is_complete(self) -> bool:
        return self.progress >= 1.0 - 1e-9


@dataclass
class ImageTask(Task):
    """Task with associated satellite image tile data.

    Extends Task with image-derived metadata for value estimation.
    Used when processing real satellite imagery through the scheduler.
    """

    tile_path: str | None = None  # Path to GeoTIFF tile
    cloud_fraction: float = 0.0  # Fraction of tile covered by clouds [0, 1]
    entropy_score: float = 0.5  # Information content / texture [0, 1]
    edge_density: float = 0.5  # Feature richness / detection potential [0, 1]

    @property
    def image_quality_multiplier(self) -> float:
        """Compute value multiplier based on image content.

        Different task types benefit from different image characteristics:
        - DETECTION: Penalize clouds, reward edge density (objects to detect)
        - ANOMALY: Penalize clouds, reward entropy (unusual patterns)
        - CLOUD_MASK: Reward clouds (more to mask = more useful)
        - COMPRESSION: Reward entropy (more information = better candidate)

        Returns:
            Multiplier in [0.1, 1.0] for task value.
        """
        cloud = self.cloud_fraction
        entropy = self.entropy_score
        edges = self.edge_density

        if self.task_type == TaskType.DETECTION:
            # Cloudy images are useless for detection
            # High edge density = more potential targets
            multiplier = (1 - cloud) * (0.3 * entropy + 0.7 * edges)
        elif self.task_type == TaskType.ANOMALY:
            # Similar to detection but entropy matters more
            multiplier = (1 - cloud) * (0.5 * entropy + 0.5 * edges)
        elif self.task_type == TaskType.CLOUD_MASK:
            # More clouds = more value for cloud masking task
            multiplier = 0.2 + 0.8 * cloud
        else:  # COMPRESSION
            # Higher entropy = better compression candidate
            multiplier = 0.3 + 0.7 * entropy

        return max(multiplier, 0.1)  # Floor at 10%

    def current_value(self, current_time: float) -> float:
        """Calculate task value with time decay and image quality.

        Extends parent method by applying image quality multiplier.
        """
        base_value = super().current_value(current_time)
        return base_value * self.image_quality_multiplier


class TaskQueue:
    """Priority queue for satellite processing tasks.

    Manages task arrivals, departures, and provides observation vectors
    for the RL agent.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._tasks: list[Task] = []

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks)

    @property
    def is_full(self) -> bool:
        return len(self._tasks) >= self.max_size

    @property
    def is_empty(self) -> bool:
        return len(self._tasks) == 0

    def add(self, task: Task) -> bool:
        """Add task to queue. Returns False if queue is full."""
        if self.is_full:
            return False
        self._tasks.append(task)
        return True

    def remove(self, task: Task) -> None:
        """Remove a specific task from the queue."""
        self._tasks.remove(task)

    def remove_completed(self) -> list[Task]:
        """Remove and return all completed tasks."""
        completed = [t for t in self._tasks if t.is_complete]
        self._tasks = [t for t in self._tasks if not t.is_complete]
        return completed

    def get_by_type(self, task_type: TaskType) -> list[Task]:
        """Get all tasks of a specific type."""
        return [t for t in self._tasks if t.task_type == task_type]

    def get_task_at_index(self, index: int) -> Task | None:
        """Get task at specific index, or None if out of bounds."""
        if 0 <= index < len(self._tasks):
            return self._tasks[index]
        return None

    def get_queue_depths(self) -> np.ndarray:
        """Return count of tasks per type as numpy array."""
        depths = np.zeros(len(TaskType), dtype=np.float32)
        for task in self._tasks:
            depths[task.task_type] += 1
        return depths

    def get_total_compute_by_type(self) -> np.ndarray:
        """Return total remaining compute per task type."""
        compute = np.zeros(len(TaskType), dtype=np.float32)
        for task in self._tasks:
            compute[task.task_type] += task.remaining_compute
        return compute

    def get_total_memory(self) -> float:
        """Return total memory footprint of queued tasks."""
        return sum(t.spec.memory_footprint for t in self._tasks)

    def get_oldest_by_type(self) -> np.ndarray:
        """Return age of oldest task per type (for urgency signal)."""
        oldest = np.zeros(len(TaskType), dtype=np.float32)
        for task_type in TaskType:
            tasks = self.get_by_type(task_type)
            if tasks:
                oldest[task_type] = max(t.arrival_time for t in tasks)
        return oldest

    def clear(self) -> None:
        """Remove all tasks from queue."""
        self._tasks.clear()
