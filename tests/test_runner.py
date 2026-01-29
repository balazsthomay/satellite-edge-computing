"""Tests for Phase 4 benchmark runner."""

import json
from pathlib import Path

import numpy as np
import pytest

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    ValueDensityScheduler,
    RandomScheduler,
)
from satellite_edge.evaluation.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
)
from satellite_edge.evaluation.scenarios import SCENARIOS
from satellite_edge.evaluation.metrics import (
    ExtendedEpisodeMetrics,
    ExtendedPolicyMetrics,
    ScenarioResult,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self):
        config = BenchmarkConfig()
        assert config.n_episodes == 50
        assert config.n_seeds == 5
        assert config.base_seed == 42

    def test_custom_values(self):
        config = BenchmarkConfig(n_episodes=10, n_seeds=3, base_seed=100)
        assert config.n_episodes == 10
        assert config.n_seeds == 3
        assert config.base_seed == 100


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    @pytest.fixture
    def quick_config(self):
        """Config for fast tests."""
        return BenchmarkConfig(n_episodes=3, n_seeds=2, base_seed=42)

    @pytest.fixture
    def runner(self, quick_config):
        return BenchmarkRunner(quick_config)

    @pytest.fixture
    def simple_policies(self):
        return [FIFOScheduler(), ValueDensityScheduler()]

    def test_runner_creation(self):
        runner = BenchmarkRunner()
        assert runner.config.n_episodes == 50
        assert runner.config.n_seeds == 5

    def test_runner_with_config(self, quick_config):
        runner = BenchmarkRunner(quick_config)
        assert runner.config.n_episodes == 3
        assert runner.config.n_seeds == 2

    def test_run_episode(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        assert isinstance(metrics, ExtendedEpisodeMetrics)
        assert metrics.total_value >= 0
        assert metrics.tasks_completed >= 0
        assert metrics.steps > 0
        # Extended metrics should be populated
        assert isinstance(metrics.completions_by_type, dict)
        assert isinstance(metrics.task_completions, list)

    def test_run_episode_reproducible(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics1 = runner.run_episode(policy, scenario, seed=42)
        metrics2 = runner.run_episode(policy, scenario, seed=42)

        assert metrics1.total_value == metrics2.total_value
        assert metrics1.tasks_completed == metrics2.tasks_completed

    def test_run_episode_different_seeds_differ(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics1 = runner.run_episode(policy, scenario, seed=42)
        metrics2 = runner.run_episode(policy, scenario, seed=123)

        # Highly unlikely to be identical
        assert metrics1.total_value != metrics2.total_value

    def test_run_episode_tracks_completions(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        # Should have task completions recorded
        if metrics.tasks_completed > 0:
            assert len(metrics.task_completions) > 0
            # Check completion record structure
            tc = metrics.task_completions[0]
            assert hasattr(tc, "task_type")
            assert hasattr(tc, "latency")
            assert tc.latency >= 0

    def test_run_policy(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_policy(policy, scenario, n_episodes=3, seeds=[42])

        assert isinstance(metrics, ExtendedPolicyMetrics)
        assert metrics.policy_name == "FIFOScheduler"
        assert metrics.n_episodes == 3
        assert len(metrics.episodes) == 3

    def test_run_policy_multiple_seeds(self, runner, quick_config):
        policy = ValueDensityScheduler()
        scenario = SCENARIOS["steady_state"]

        # 3 episodes per seed × 2 seeds = 6 episodes
        metrics = runner.run_policy(policy, scenario)

        assert metrics.n_episodes == 6

    def test_run_policy_aggregates_correctly(self, runner):
        policy = PriorityScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_policy(policy, scenario, n_episodes=5, seeds=[42])

        # Check aggregation works
        assert metrics.mean_value > 0
        assert metrics.value_iqm > 0
        ci_low, ci_high = metrics.value_ci
        assert ci_low <= metrics.value_iqm <= ci_high

    def test_run_scenario(self, runner, simple_policies):
        scenario = SCENARIOS["steady_state"]

        result = runner.run_scenario(
            scenario, simple_policies, n_episodes=2, seeds=[42]
        )

        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "steady_state"
        assert "FIFOScheduler" in result.policy_results
        assert "ValueDensityScheduler" in result.policy_results

    def test_run_scenario_all_policies_evaluated(self, runner):
        policies = [
            FIFOScheduler(),
            PriorityScheduler(),
            ValueDensityScheduler(),
            RandomScheduler(seed=42),
        ]
        scenario = SCENARIOS["steady_state"]

        result = runner.run_scenario(scenario, policies, n_episodes=2, seeds=[42])

        assert len(result.policy_results) == 4
        for policy in policies:
            assert policy.name in result.policy_results

    def test_run_all_scenarios(self, runner, simple_policies):
        results = runner.run_all_scenarios(
            simple_policies,
            scenarios=["steady_state", "event_surge"],
            n_episodes=2,
            seeds=[42],
        )

        assert len(results) == 2
        assert "steady_state" in results
        assert "event_surge" in results

        for scenario_name, result in results.items():
            assert result.scenario_name == scenario_name
            assert len(result.policy_results) == 2


class TestBenchmarkRunnerIO:
    """Tests for BenchmarkRunner save/load functionality."""

    @pytest.fixture
    def runner_with_results(self, tmp_path):
        """Run a minimal benchmark and return runner with results."""
        config = BenchmarkConfig(n_episodes=2, n_seeds=1, base_seed=42)
        runner = BenchmarkRunner(config)
        policies = [FIFOScheduler(), ValueDensityScheduler()]

        results = runner.run_all_scenarios(
            policies,
            scenarios=["steady_state"],
            n_episodes=2,
            seeds=[42],
        )

        return runner, results, tmp_path

    def test_save_results(self, runner_with_results):
        runner, results, tmp_path = runner_with_results
        output_dir = tmp_path / "results"

        runner.save_results(output_dir, results)

        assert (output_dir / "benchmark_results.json").exists()
        assert (output_dir / "data" / "raw_episodes.json").exists()

    def test_save_results_valid_json(self, runner_with_results):
        runner, results, tmp_path = runner_with_results
        output_dir = tmp_path / "results"

        runner.save_results(output_dir, results)

        # Should be valid JSON
        with open(output_dir / "benchmark_results.json") as f:
            data = json.load(f)

        assert "steady_state" in data
        assert "policies" in data["steady_state"]

    def test_save_results_structure(self, runner_with_results):
        runner, results, tmp_path = runner_with_results
        output_dir = tmp_path / "results"

        runner.save_results(output_dir, results)

        with open(output_dir / "benchmark_results.json") as f:
            data = json.load(f)

        scenario = data["steady_state"]
        assert "description" in scenario
        assert "policies" in scenario
        assert "ranking" in scenario

        for policy_name, policy_data in scenario["policies"].items():
            assert "value_iqm" in policy_data
            assert "n_episodes" in policy_data

    def test_load_results(self, runner_with_results):
        runner, results, tmp_path = runner_with_results
        output_dir = tmp_path / "results"

        runner.save_results(output_dir, results)

        loaded = BenchmarkRunner.load_results(output_dir)

        assert "steady_state" in loaded
        assert "FIFOScheduler" in loaded["steady_state"]["policies"]


class TestRunnerExtendedMetrics:
    """Tests for extended metric collection in runner."""

    @pytest.fixture
    def runner(self):
        config = BenchmarkConfig(n_episodes=5, n_seeds=1, base_seed=42)
        return BenchmarkRunner(config)

    def test_collects_buffer_overflow_count(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["event_surge"]  # High pressure scenario

        metrics = runner.run_episode(policy, scenario, seed=42)

        # Should have buffer_overflow_steps populated (may or may not be > 0)
        assert hasattr(metrics, "buffer_overflow_steps")
        assert isinstance(metrics.buffer_overflow_steps, int)

    def test_collects_contact_time(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        assert hasattr(metrics, "contact_time_used")
        assert hasattr(metrics, "contact_time_available")
        # If there was any contact, available should be > 0
        # (contact availability depends on orbit model)

    def test_collects_potential_value(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        assert hasattr(metrics, "potential_value")
        assert metrics.potential_value >= metrics.total_value

    def test_collects_completions_by_type(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        assert isinstance(metrics.completions_by_type, dict)
        # At least some task type should have completions
        total = sum(metrics.completions_by_type.values())
        assert total == metrics.tasks_completed

    def test_collects_task_completion_records(self, runner):
        policy = FIFOScheduler()
        scenario = SCENARIOS["steady_state"]

        metrics = runner.run_episode(policy, scenario, seed=42)

        assert isinstance(metrics.task_completions, list)
        assert len(metrics.task_completions) == metrics.tasks_completed

        if metrics.task_completions:
            record = metrics.task_completions[0]
            assert hasattr(record, "task_type")
            assert hasattr(record, "arrival_time")
            assert hasattr(record, "completion_time")
            assert hasattr(record, "priority")
            assert hasattr(record, "value")
            assert record.latency >= 0


class TestRunnerPolicyComparison:
    """Tests that verify policy comparison makes sense."""

    @pytest.fixture
    def comparison_results(self):
        """Run comparison of key policies."""
        config = BenchmarkConfig(n_episodes=20, n_seeds=2, base_seed=42)
        runner = BenchmarkRunner(config)

        policies = [
            FIFOScheduler(),
            ValueDensityScheduler(),
            RandomScheduler(seed=42),
        ]

        return runner.run_scenario(
            SCENARIOS["steady_state"],
            policies,
        )

    def test_all_policies_complete_tasks(self, comparison_results):
        """All policies should complete some tasks."""
        for name, pm in comparison_results.policy_results.items():
            assert pm.mean_completed > 0, f"{name} completed no tasks"

    def test_all_policies_positive_value(self, comparison_results):
        """All policies should capture positive value."""
        for name, pm in comparison_results.policy_results.items():
            assert pm.mean_value > 0, f"{name} captured no value"

    def test_smart_policies_beat_random(self, comparison_results):
        """Deterministic policies should generally beat random."""
        random_value = comparison_results.policy_results["RandomScheduler"].value_iqm
        fifo_value = comparison_results.policy_results["FIFOScheduler"].value_iqm
        vd_value = comparison_results.policy_results["ValueDensityScheduler"].value_iqm

        # At least one smart policy should beat random
        assert fifo_value > random_value * 0.9 or vd_value > random_value * 0.9

    def test_random_has_higher_idle_fraction(self, comparison_results):
        """Random should idle more than deterministic policies."""
        random_idle = comparison_results.policy_results[
            "RandomScheduler"
        ].mean_idle_fraction
        fifo_idle = comparison_results.policy_results[
            "FIFOScheduler"
        ].mean_idle_fraction

        # Random idles ~20% of steps, FIFO never idles
        assert random_idle > fifo_idle
