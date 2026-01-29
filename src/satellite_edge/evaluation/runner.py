"""Benchmark runner for systematic policy evaluation.

Provides multi-seed, multi-scenario evaluation with proper statistical
methodology following rliable (NeurIPS 2021) best practices.
"""

from dataclasses import dataclass
from typing import Any, Protocol
import json
from pathlib import Path

import numpy as np

from satellite_edge.evaluation.metrics import (
    TaskCompletionRecord,
    ExtendedEpisodeMetrics,
    ExtendedPolicyMetrics,
    ScenarioResult,
)
from satellite_edge.evaluation.scenarios import (
    ScenarioConfig,
    SCENARIOS,
    create_scenario_env,
)
from satellite_edge.agents.baselines import BaseScheduler


class PolicyProtocol(Protocol):
    """Protocol for policies that can be evaluated."""

    @property
    def name(self) -> str: ...

    def select_action(self, obs: np.ndarray) -> int: ...

    def reset(self) -> None: ...


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    n_episodes: int = 50  # Episodes per seed
    n_seeds: int = 5  # Seeds for statistical validity
    base_seed: int = 42  # Starting seed for reproducibility


class BenchmarkRunner:
    """Orchestrates benchmark evaluation across scenarios and policies.

    Implements proper statistical methodology:
    - Multiple seeds for variance estimation
    - IQM aggregation for robustness
    - Bootstrap CIs for uncertainty quantification
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self._results: dict[str, ScenarioResult] = {}

    def run_episode(
        self,
        policy: PolicyProtocol,
        scenario: ScenarioConfig,
        seed: int,
    ) -> ExtendedEpisodeMetrics:
        """Run a single evaluation episode.

        Args:
            policy: Policy to evaluate.
            scenario: Scenario configuration.
            seed: Random seed for this episode.

        Returns:
            Extended metrics from the episode.
        """
        env = create_scenario_env(scenario, seed=seed)
        obs, info = env.reset(seed=seed)
        policy.reset()

        metrics = ExtendedEpisodeMetrics()
        queue_sizes: list[float] = []
        num_task_types = 4  # len(TaskType)

        done = False
        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            metrics.total_reward += reward
            metrics.steps += 1

            if action == num_task_types:
                metrics.idle_steps += 1

            queue_sizes.append(info["queue_size"])

        # Extract final metrics
        metrics.total_value = info["total_value"]
        metrics.tasks_completed = info["tasks_completed"]
        metrics.tasks_dropped = info["tasks_dropped"]
        metrics.downlink_data = info["downlink_data"]
        metrics.queue_occupancy_mean = (
            float(np.mean(queue_sizes)) if queue_sizes else 0.0
        )

        # Extended metrics
        metrics.buffer_overflow_steps = info["buffer_overflows"]
        metrics.contact_time_used = info["contact_time_used"]
        metrics.contact_time_available = info["contact_time_available"]
        metrics.potential_value = info["potential_value"]
        metrics.completions_by_type = info["completions_by_type"]
        metrics.value_by_type = info["value_by_type"]

        # Convert task completion records
        for record in info["task_completions"]:
            metrics.task_completions.append(
                TaskCompletionRecord(
                    task_type=record["task_type"],
                    arrival_time=record["arrival_time"],
                    completion_time=record["completion_time"],
                    priority=record["priority"],
                    value=record["value"],
                )
            )

        env.close()
        return metrics

    def run_policy(
        self,
        policy: PolicyProtocol,
        scenario: ScenarioConfig,
        n_episodes: int | None = None,
        seeds: list[int] | None = None,
    ) -> ExtendedPolicyMetrics:
        """Evaluate a policy over multiple episodes and seeds.

        Args:
            policy: Policy to evaluate.
            scenario: Scenario configuration.
            n_episodes: Episodes per seed (default from config).
            seeds: List of seeds (default from config).

        Returns:
            Aggregated policy metrics.
        """
        n_episodes = n_episodes or self.config.n_episodes
        if seeds is None:
            seeds = [
                self.config.base_seed + i * 1000 for i in range(self.config.n_seeds)
            ]

        policy_metrics = ExtendedPolicyMetrics(policy_name=policy.name)

        for seed in seeds:
            for ep in range(n_episodes):
                episode_seed = seed + ep
                ep_metrics = self.run_episode(policy, scenario, episode_seed)
                policy_metrics.episodes.append(ep_metrics)

        return policy_metrics

    def run_scenario(
        self,
        scenario: ScenarioConfig,
        policies: list[PolicyProtocol],
        n_episodes: int | None = None,
        seeds: list[int] | None = None,
    ) -> ScenarioResult:
        """Evaluate all policies on a single scenario.

        Args:
            scenario: Scenario to evaluate.
            policies: List of policies to compare.
            n_episodes: Episodes per seed.
            seeds: List of seeds.

        Returns:
            ScenarioResult with all policy evaluations.
        """
        result = ScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
        )

        for policy in policies:
            policy_metrics = self.run_policy(policy, scenario, n_episodes, seeds)
            result.add_policy_result(policy_metrics)

        return result

    def run_all_scenarios(
        self,
        policies: list[PolicyProtocol],
        scenarios: list[str] | None = None,
        n_episodes: int | None = None,
        seeds: list[int] | None = None,
    ) -> dict[str, ScenarioResult]:
        """Evaluate all policies across all scenarios.

        Args:
            policies: List of policies to compare.
            scenarios: Scenario names to run (default: all).
            n_episodes: Episodes per seed.
            seeds: List of seeds.

        Returns:
            Dict mapping scenario names to ScenarioResults.
        """
        if scenarios is None:
            scenarios = list(SCENARIOS.keys())

        results = {}
        for scenario_name in scenarios:
            scenario = SCENARIOS[scenario_name]
            results[scenario_name] = self.run_scenario(
                scenario, policies, n_episodes, seeds
            )

        self._results = results
        return results

    def save_results(
        self,
        output_dir: str | Path,
        results: dict[str, ScenarioResult] | None = None,
    ) -> None:
        """Save benchmark results to disk.

        Args:
            output_dir: Directory to save results.
            results: Results to save (default: last run results).
        """
        results = results or self._results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Save summary JSON
        summary = {}
        for scenario_name, scenario_result in results.items():
            summary[scenario_name] = {
                "description": scenario_result.description,
                "policies": {
                    name: pm.summary()
                    for name, pm in scenario_result.policy_results.items()
                },
                "ranking": scenario_result.get_ranking(),
            }

        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_serializer)

        # Save raw episode data for detailed analysis
        raw_data = {}
        for scenario_name, scenario_result in results.items():
            raw_data[scenario_name] = {}
            for policy_name, pm in scenario_result.policy_results.items():
                raw_data[scenario_name][policy_name] = [
                    _episode_to_dict(ep) for ep in pm.episodes
                ]

        with open(data_dir / "raw_episodes.json", "w") as f:
            json.dump(raw_data, f, indent=2, default=_json_serializer)

    @staticmethod
    def load_results(input_dir: str | Path) -> dict[str, Any]:
        """Load benchmark results from disk.

        Args:
            input_dir: Directory containing results.

        Returns:
            Loaded results dictionary.
        """
        input_dir = Path(input_dir)
        with open(input_dir / "benchmark_results.json") as f:
            return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _episode_to_dict(ep: ExtendedEpisodeMetrics) -> dict[str, Any]:
    """Convert episode metrics to JSON-serializable dict."""
    return {
        "total_value": ep.total_value,
        "total_reward": ep.total_reward,
        "tasks_completed": ep.tasks_completed,
        "tasks_dropped": ep.tasks_dropped,
        "buffer_overflow_steps": ep.buffer_overflow_steps,
        "downlink_data": ep.downlink_data,
        "idle_steps": ep.idle_steps,
        "queue_occupancy_mean": ep.queue_occupancy_mean,
        "steps": ep.steps,
        "contact_time_used": ep.contact_time_used,
        "contact_time_available": ep.contact_time_available,
        "potential_value": ep.potential_value,
        "completions_by_type": ep.completions_by_type,
        "value_by_type": ep.value_by_type,
        "latency_percentiles": ep.latency_percentiles(),
        "n_task_completions": len(ep.task_completions),
    }
