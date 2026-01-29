"""Extended metrics for rigorous policy evaluation.

Implements metrics from rliable best practices (NeurIPS 2021):
- IQM (interquartile mean) for robustness to outliers
- Stratified bootstrap 95% CIs
- Per-task-type analysis for fairness evaluation
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskCompletionRecord:
    """Record of a single task completion for latency analysis.

    Captures timing and value information for post-hoc analysis of
    scheduling quality, particularly latency distributions.
    """

    task_type: int  # TaskType.value
    arrival_time: float  # Simulation time when task entered queue
    completion_time: float  # Simulation time when task completed
    priority: float  # Priority boost factor (1.0 = normal)
    value: float  # Captured value at completion

    @property
    def latency(self) -> float:
        """Time from arrival to completion in seconds."""
        return self.completion_time - self.arrival_time


@dataclass
class ExtendedEpisodeMetrics:
    """Extended metrics from a single evaluation episode.

    Adds latency distributions, per-type completions, and contact efficiency
    to the basic EpisodeMetrics from agents.evaluation.
    """

    # Core metrics
    total_value: float = 0.0
    total_reward: float = 0.0
    tasks_completed: int = 0
    tasks_dropped: int = 0
    buffer_overflow_steps: int = 0
    downlink_data: float = 0.0
    idle_steps: int = 0
    queue_occupancy_mean: float = 0.0
    steps: int = 0

    # Extended metrics
    task_completions: list[TaskCompletionRecord] = field(default_factory=list)
    completions_by_type: dict[int, int] = field(default_factory=dict)
    value_by_type: dict[int, float] = field(default_factory=dict)
    contact_time_used: float = 0.0
    contact_time_available: float = 0.0
    potential_value: float = 0.0  # Value if all tasks completed instantly

    @property
    def contact_efficiency(self) -> float:
        """Fraction of contact window utilized for downlink."""
        if self.contact_time_available < 1e-6:
            return 0.0
        return self.contact_time_used / self.contact_time_available

    @property
    def buffer_overflow_rate(self) -> float:
        """Fraction of steps with buffer overflow."""
        if self.steps == 0:
            return 0.0
        return self.buffer_overflow_steps / self.steps

    @property
    def value_capture_efficiency(self) -> float:
        """Captured value / potential value (accounts for decay)."""
        if self.potential_value < 1e-6:
            return 0.0
        return self.total_value / self.potential_value

    def latency_percentiles(self) -> dict[str, float]:
        """Compute latency percentiles (P50, P95, P99)."""
        if not self.task_completions:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        latencies = [tc.latency for tc in self.task_completions]
        return {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        }

    def priority_task_latencies(self) -> list[float]:
        """Get latencies for high-priority tasks (priority > 1.5)."""
        return [tc.latency for tc in self.task_completions if tc.priority > 1.5]

    def latencies_by_type(self) -> dict[int, list[float]]:
        """Get latencies grouped by task type."""
        result: dict[int, list[float]] = {}
        for tc in self.task_completions:
            if tc.task_type not in result:
                result[tc.task_type] = []
            result[tc.task_type].append(tc.latency)
        return result


def compute_iqm(values: list[float]) -> float:
    """Compute interquartile mean (IQM).

    IQM is robust to outliers and provides a more stable aggregate
    metric than simple mean. Recommended by rliable (NeurIPS 2021).

    Args:
        values: List of metric values across episodes/seeds.

    Returns:
        Interquartile mean, or 0.0 if insufficient data.
    """
    if len(values) < 4:
        return float(np.mean(values)) if values else 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    return float(np.mean(sorted_vals[q1_idx:q3_idx]))


def compute_bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Uses stratified bootstrap with IQM as the estimator for robustness.

    Args:
        values: List of metric values.
        confidence: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap replications.
        rng: Random generator for reproducibility.

    Returns:
        (lower_bound, upper_bound) of confidence interval.
    """
    if len(values) < 2:
        val = values[0] if values else 0.0
        return (val, val)

    rng = rng or np.random.default_rng()
    arr = np.array(values)
    bootstrap_iqms = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_iqms.append(compute_iqm(sample.tolist()))

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_iqms, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_iqms, 100 * (1 - alpha / 2)))
    return (lower, upper)


def compute_jains_fairness(values: list[float]) -> float:
    """Compute Jain's fairness index.

    Measures equitable distribution of resources (task completions) across
    task types. Ranges from 1/n (completely unfair) to 1 (perfectly fair).

    J(x) = (sum(x_i))^2 / (n * sum(x_i^2))

    Args:
        values: Completion counts or rates per task type.

    Returns:
        Jain's fairness index in [0, 1].
    """
    if not values or all(v == 0 for v in values):
        return 0.0

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    sum_x = np.sum(arr)
    sum_x2 = np.sum(arr**2)

    if sum_x2 < 1e-9:
        return 1.0  # All zeros treated as perfectly fair

    return float(sum_x**2 / (n * sum_x2))


@dataclass
class ExtendedPolicyMetrics:
    """Aggregated metrics across multiple episodes with statistical rigor.

    Extends PolicyMetrics with IQM aggregates and bootstrap CIs.
    """

    policy_name: str
    episodes: list[ExtendedEpisodeMetrics] = field(default_factory=list)
    _bootstrap_rng_seed: int = 42

    # Cached statistics (computed lazily)
    _value_iqm: float | None = field(default=None, repr=False)
    _value_ci: tuple[float, float] | None = field(default=None, repr=False)

    def _values(self) -> list[float]:
        return [e.total_value for e in self.episodes]

    def _rewards(self) -> list[float]:
        return [e.total_reward for e in self.episodes]

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def value_iqm(self) -> float:
        """Interquartile mean of total value."""
        if self._value_iqm is None:
            self._value_iqm = compute_iqm(self._values())
        return self._value_iqm

    @property
    def value_ci(self) -> tuple[float, float]:
        """95% bootstrap CI for value IQM."""
        if self._value_ci is None:
            rng = np.random.default_rng(self._bootstrap_rng_seed)
            self._value_ci = compute_bootstrap_ci(self._values(), rng=rng)
        return self._value_ci

    @property
    def reward_iqm(self) -> float:
        """Interquartile mean of total reward."""
        return compute_iqm(self._rewards())

    @property
    def mean_value(self) -> float:
        """Simple mean for backward compatibility."""
        if not self.episodes:
            return 0.0
        return float(np.mean(self._values()))

    @property
    def std_value(self) -> float:
        """Standard deviation of value."""
        if not self.episodes:
            return 0.0
        return float(np.std(self._values()))

    @property
    def mean_completed(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tasks_completed for e in self.episodes]))

    @property
    def mean_dropped(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tasks_dropped for e in self.episodes]))

    @property
    def mean_buffer_overflow_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.buffer_overflow_rate for e in self.episodes]))

    @property
    def mean_contact_efficiency(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.contact_efficiency for e in self.episodes]))

    @property
    def mean_idle_fraction(self) -> float:
        if not self.episodes:
            return 0.0
        fractions = [e.idle_steps / max(e.steps, 1) for e in self.episodes]
        return float(np.mean(fractions))

    def latency_p95_iqm(self) -> float:
        """IQM of P95 latencies across episodes."""
        p95s = [e.latency_percentiles()["p95"] for e in self.episodes]
        return compute_iqm(p95s)

    def latency_p95_ci(self) -> tuple[float, float]:
        """95% CI for P95 latency IQM."""
        p95s = [e.latency_percentiles()["p95"] for e in self.episodes]
        rng = np.random.default_rng(self._bootstrap_rng_seed)
        return compute_bootstrap_ci(p95s, rng=rng)

    def priority_latency_iqm(self) -> float:
        """IQM of mean latency for high-priority tasks."""
        means = []
        for e in self.episodes:
            lats = e.priority_task_latencies()
            if lats:
                means.append(float(np.mean(lats)))
        return compute_iqm(means) if means else 0.0

    def jains_fairness(self) -> float:
        """Jain's fairness index for task type completions.

        Computed per-episode then averaged.
        """
        if not self.episodes:
            return 0.0

        fairness_scores = []
        for e in self.episodes:
            if e.completions_by_type:
                counts = list(e.completions_by_type.values())
                fairness_scores.append(compute_jains_fairness(counts))
        return float(np.mean(fairness_scores)) if fairness_scores else 0.0

    def completion_rates_by_type(self) -> dict[int, float]:
        """Mean completion count per task type across episodes."""
        if not self.episodes:
            return {}

        type_totals: dict[int, list[float]] = {}
        for e in self.episodes:
            for task_type, count in e.completions_by_type.items():
                if task_type not in type_totals:
                    type_totals[task_type] = []
                type_totals[task_type].append(count)

        return {t: float(np.mean(counts)) for t, counts in type_totals.items()}

    def summary(self) -> dict[str, Any]:
        """Return comprehensive summary dictionary."""
        ci_low, ci_high = self.value_ci
        p95_ci_low, p95_ci_high = self.latency_p95_ci()

        return {
            "policy": self.policy_name,
            "n_episodes": self.n_episodes,
            # Value metrics
            "value_iqm": self.value_iqm,
            "value_ci_low": ci_low,
            "value_ci_high": ci_high,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            # Task metrics
            "mean_completed": self.mean_completed,
            "mean_dropped": self.mean_dropped,
            "jains_fairness": self.jains_fairness(),
            # Latency metrics
            "latency_p95_iqm": self.latency_p95_iqm(),
            "latency_p95_ci_low": p95_ci_low,
            "latency_p95_ci_high": p95_ci_high,
            "priority_latency_iqm": self.priority_latency_iqm(),
            # Efficiency metrics
            "mean_buffer_overflow_rate": self.mean_buffer_overflow_rate,
            "mean_contact_efficiency": self.mean_contact_efficiency,
            "mean_idle_fraction": self.mean_idle_fraction,
            # Per-type breakdown
            "completion_rates_by_type": self.completion_rates_by_type(),
        }


@dataclass
class ScenarioResult:
    """Results from evaluating multiple policies on a single scenario."""

    scenario_name: str
    description: str
    policy_results: dict[str, ExtendedPolicyMetrics] = field(default_factory=dict)

    def add_policy_result(self, metrics: ExtendedPolicyMetrics) -> None:
        """Add results for a policy."""
        self.policy_results[metrics.policy_name] = metrics

    def get_ranking(self, metric: str = "value_iqm") -> list[tuple[str, float]]:
        """Get policies ranked by specified metric (descending).

        Args:
            metric: Metric to rank by (default: value_iqm).

        Returns:
            List of (policy_name, metric_value) tuples, sorted descending.
        """
        rankings = []
        for name, pm in self.policy_results.items():
            summary = pm.summary()
            val = summary.get(metric, 0.0)
            rankings.append((name, val))

        # Higher is better for most metrics except latency
        reverse = "latency" not in metric.lower()
        return sorted(rankings, key=lambda x: x[1], reverse=reverse)

    def summary_table(self) -> list[dict[str, Any]]:
        """Generate summary table for all policies."""
        return [pm.summary() for pm in self.policy_results.values()]
