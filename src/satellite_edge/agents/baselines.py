"""Baseline scheduling policies for comparison against RL agents.

Implements FIFO, Priority-based, Round-Robin, and Greedy-Compute strategies.
All baselines implement the same interface for fair comparison.
"""

from abc import ABC, abstractmethod

import numpy as np

from satellite_edge.environment.tasks import TaskType, TASK_SPECS


class BaseScheduler(ABC):
    """Abstract base class for scheduling policies.

    All schedulers operate purely on the observation vector returned by
    SatelliteEnv, making them compatible with any evaluation pipeline.
    """

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Select an action given the current observation.

        Args:
            obs: 14-dimensional observation vector from SatelliteEnv.

        Returns:
            Action index (0-3 for task types, 4 for IDLE).
        """

    def reset(self) -> None:
        """Reset any internal state (e.g., for round-robin cycling)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class FIFOScheduler(BaseScheduler):
    """First-In-First-Out: prioritize the task type with deepest queue.

    Approximates true FIFO by focusing compute on whichever task type
    has accumulated the most backlog (highest queue depth).
    This is the standard "do the oldest work first" baseline.
    """

    def select_action(self, obs: np.ndarray) -> int:
        # Queue depths are obs[2:6], normalized by max_size (50)
        queue_depths = obs[2:6]

        if np.sum(queue_depths) < 1e-6:
            return len(TaskType)  # IDLE

        # Pick the deepest queue (oldest work accumulates here)
        return int(np.argmax(queue_depths))


class PriorityScheduler(BaseScheduler):
    """Static Priority: always prioritize the highest-value task type.

    Uses the base_value * deadline_sensitivity product to determine
    static priority ordering. ANOMALY > DETECTION > COMPRESSION > CLOUD_MASK.
    A strong baseline when one task type dominates value generation.
    """

    def __init__(self):
        # Pre-compute priority ranking: value * urgency
        self._priority_order = sorted(
            TaskType,
            key=lambda t: TASK_SPECS[t].base_value * TASK_SPECS[t].deadline_sensitivity,
            reverse=True,
        )

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]

        # Pick highest-priority type that has tasks
        for task_type in self._priority_order:
            if queue_depths[task_type] > 1e-6:
                return int(task_type)

        return len(TaskType)  # IDLE


class RoundRobinScheduler(BaseScheduler):
    """Round-Robin: cycle through task types equally.

    Fair allocation of compute across task types. Provides a baseline
    for "no intelligence, just fairness" scheduling.
    """

    def __init__(self):
        self._current: int = 0

    def reset(self) -> None:
        self._current = 0

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]
        num_types = len(TaskType)

        # Try each type starting from current position
        for _ in range(num_types):
            task_type = self._current % num_types
            self._current += 1
            if queue_depths[task_type] > 1e-6:
                return task_type

        return num_types  # IDLE


class GreedyComputeScheduler(BaseScheduler):
    """Greedy-Compute: prioritize the cheapest tasks (Shortest Job First).

    Maximizes throughput by focusing on tasks that complete fastest,
    giving the most task completions per unit compute. Ignores value.
    """

    def __init__(self):
        # Pre-compute cost ordering
        self._cost_order = sorted(
            TaskType,
            key=lambda t: TASK_SPECS[t].compute_cost,
        )

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]

        # Pick cheapest type that has tasks
        for task_type in self._cost_order:
            if queue_depths[task_type] > 1e-6:
                return int(task_type)

        return len(TaskType)  # IDLE


class RandomScheduler(BaseScheduler):
    """Random: uniform random action selection.

    Lower bound baseline. Any reasonable policy should beat this.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, len(TaskType) + 1))
