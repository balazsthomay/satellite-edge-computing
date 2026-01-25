"""Baseline scheduling policies for comparison against RL agents.

Implements FIFO, Priority-based, Round-Robin, Greedy-Compute, and Value-Density
strategies. All baselines implement the same interface for fair comparison.

Key design principle: NEVER IDLE. The observation reflects queue state AFTER
processing (often empty), but new Poisson arrivals occur at step start. With
arrival_rate=3.0, P(zero arrivals) ≈ 5%. Idling wastes 95% of opportunities.
Non-IDLE action with empty queue is harmless (processing loop does nothing).
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

    Never idles: defaults to DETECTION (highest arrival weight ≈ 0.35)
    when all queues appear empty in the observation.
    """

    # DETECTION has highest arrival weight in default config
    _DEFAULT_TYPE = int(TaskType.DETECTION)

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]

        if np.sum(queue_depths) < 1e-6:
            return self._DEFAULT_TYPE  # Never idle

        return int(np.argmax(queue_depths))


class PriorityScheduler(BaseScheduler):
    """Static Priority: prioritize by queue_depth × value_rate.

    Value rates (base_value × deadline_sensitivity):
      ANOMALY=13.5, DETECTION=8.0, COMPRESSION=2.5, CLOUD_MASK=0.9

    When queues appear empty, defaults to ANOMALY (highest value rate).
    This captures the most time-sensitive value under contention.
    """

    def __init__(self):
        # Pre-compute value rates: base_value * deadline_sensitivity
        self._value_rates = np.array([
            TASK_SPECS[t].base_value * TASK_SPECS[t].deadline_sensitivity
            for t in TaskType
        ], dtype=np.float32)
        # ANOMALY has highest value rate
        self._default_type = int(TaskType.ANOMALY)

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]

        if np.sum(queue_depths) < 1e-6:
            return self._default_type  # Never idle

        # Weighted priority: depth × value_rate
        weighted = queue_depths * self._value_rates
        return int(np.argmax(weighted))


class RoundRobinScheduler(BaseScheduler):
    """Round-Robin: cycle through task types unconditionally.

    Fair allocation of compute across task types. Never idles —
    always picks the next type in sequence regardless of observed
    queue state (arrivals happen before processing).
    """

    def __init__(self):
        self._current: int = 0

    def reset(self) -> None:
        self._current = 0

    def select_action(self, obs: np.ndarray) -> int:
        # Never idle — just cycle through types unconditionally
        task_type = self._current % len(TaskType)
        self._current += 1
        return task_type


class GreedyComputeScheduler(BaseScheduler):
    """Greedy-Compute: prioritize by total pending compute (obs[6:10]).

    Focuses on reducing compute backlog by selecting the task type
    with the most accumulated compute work. This maximizes throughput
    when the bottleneck is processing capacity.

    Defaults to DETECTION when empty (most compute-heavy tasks).
    """

    _DEFAULT_TYPE = int(TaskType.DETECTION)

    def select_action(self, obs: np.ndarray) -> int:
        queue_compute = obs[6:10]

        if np.sum(queue_compute) < 1e-6:
            return self._DEFAULT_TYPE  # Never idle

        return int(np.argmax(queue_compute))


class ValueDensityScheduler(BaseScheduler):
    """Value-Density: maximize value per TOPS.

    Selects task type with highest (value × sensitivity / compute_cost).
    This is the strongest heuristic—maximizes expected value capture
    per unit of compute spent.

    Density scores:
      ANOMALY=4.5, DETECTION=1.33, COMPRESSION=0.625, CLOUD_MASK=0.6

    When queues appear empty, picks ANOMALY (highest density).
    Robust default since ANOMALY tasks have highest value/compute ratio.
    """

    def __init__(self):
        # Pre-compute value density: (base_value * sensitivity) / compute_cost
        self._density = np.array([
            (TASK_SPECS[t].base_value * TASK_SPECS[t].deadline_sensitivity)
            / TASK_SPECS[t].compute_cost
            for t in TaskType
        ], dtype=np.float32)
        # Priority order by density (descending)
        self._priority_order = list(np.argsort(-self._density))
        # ANOMALY has highest density
        self._default_type = int(TaskType.ANOMALY)

    def select_action(self, obs: np.ndarray) -> int:
        queue_depths = obs[2:6]

        if np.sum(queue_depths) < 1e-6:
            return self._default_type  # Never idle

        # Pick highest-density type that has tasks
        for task_type in self._priority_order:
            if queue_depths[task_type] > 1e-6:
                return task_type

        return self._default_type


class RandomScheduler(BaseScheduler):
    """Random: uniform random action selection (includes IDLE).

    Lower bound baseline. Any reasonable policy should beat this.
    Includes IDLE in action space, so it naturally idles ~20% of steps.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, len(TaskType) + 1))
