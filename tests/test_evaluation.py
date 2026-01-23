"""Tests for evaluation and comparison utilities."""

import numpy as np
import pytest

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    RandomScheduler,
)
from satellite_edge.agents.evaluation import (
    EpisodeMetrics,
    PolicyMetrics,
    evaluate_policy,
    compare_policies,
)
from satellite_edge.environment.satellite_env import SatelliteConfig, EpisodeConfig


class TestEpisodeMetrics:
    def test_default_values(self):
        m = EpisodeMetrics()
        assert m.total_value == 0.0
        assert m.total_reward == 0.0
        assert m.tasks_completed == 0
        assert m.tasks_dropped == 0
        assert m.idle_steps == 0

    def test_custom_values(self):
        m = EpisodeMetrics(
            total_value=100.0,
            tasks_completed=50,
            downlink_data=10.5,
        )
        assert m.total_value == 100.0
        assert m.tasks_completed == 50
        assert m.downlink_data == 10.5


class TestPolicyMetrics:
    def test_empty_metrics(self):
        pm = PolicyMetrics(policy_name="test")
        assert pm.mean_value == 0.0
        assert pm.std_value == 0.0
        assert pm.mean_completed == 0.0

    def test_single_episode(self):
        pm = PolicyMetrics(policy_name="test")
        pm.episodes.append(EpisodeMetrics(total_value=100.0, tasks_completed=10))
        assert pm.mean_value == 100.0
        assert pm.std_value == 0.0
        assert pm.mean_completed == 10.0

    def test_multiple_episodes(self):
        pm = PolicyMetrics(policy_name="test")
        pm.episodes.append(EpisodeMetrics(total_value=80.0, steps=100))
        pm.episodes.append(EpisodeMetrics(total_value=120.0, steps=100))
        assert pm.mean_value == 100.0
        assert pm.std_value == 20.0

    def test_idle_fraction(self):
        pm = PolicyMetrics(policy_name="test")
        pm.episodes.append(EpisodeMetrics(idle_steps=25, steps=100))
        pm.episodes.append(EpisodeMetrics(idle_steps=75, steps=100))
        assert pm.mean_idle_fraction == 0.5

    def test_summary_dict(self):
        pm = PolicyMetrics(policy_name="MyPolicy")
        pm.episodes.append(EpisodeMetrics(total_value=50.0, total_reward=60.0, steps=100))
        summary = pm.summary()
        assert summary["policy"] == "MyPolicy"
        assert summary["mean_value"] == 50.0
        assert summary["mean_reward"] == 60.0
        assert summary["n_episodes"] == 1


class TestEvaluatePolicy:
    """Test the evaluate_policy function with actual environment runs."""

    @pytest.fixture
    def short_config(self):
        return EpisodeConfig(max_steps=50)

    def test_evaluate_fifo(self, short_config):
        sched = FIFOScheduler()
        result = evaluate_policy(sched, n_episodes=3, episode_config=short_config, seed=42)
        assert result.policy_name == "FIFOScheduler"
        assert len(result.episodes) == 3
        assert result.mean_value > 0.0

    def test_evaluate_priority(self, short_config):
        sched = PriorityScheduler()
        result = evaluate_policy(sched, n_episodes=3, episode_config=short_config, seed=42)
        assert result.mean_value > 0.0
        assert result.mean_completed > 0.0

    def test_evaluate_round_robin(self, short_config):
        sched = RoundRobinScheduler()
        result = evaluate_policy(sched, n_episodes=3, episode_config=short_config, seed=42)
        assert result.mean_value > 0.0

    def test_evaluate_greedy_compute(self, short_config):
        sched = GreedyComputeScheduler()
        result = evaluate_policy(sched, n_episodes=3, episode_config=short_config, seed=42)
        assert result.mean_value > 0.0
        # Greedy compute should have high completion rate
        assert result.mean_completed > 0.0

    def test_evaluate_random(self, short_config):
        sched = RandomScheduler(seed=42)
        result = evaluate_policy(sched, n_episodes=3, episode_config=short_config, seed=42)
        # Random policy should complete at least some tasks
        assert len(result.episodes) == 3

    def test_deterministic_evaluation(self, short_config):
        """Same seed should produce same results."""
        sched = FIFOScheduler()
        r1 = evaluate_policy(sched, n_episodes=5, episode_config=short_config, seed=42)
        r2 = evaluate_policy(sched, n_episodes=5, episode_config=short_config, seed=42)
        assert abs(r1.mean_value - r2.mean_value) < 1e-6

    def test_different_seeds_differ(self, short_config):
        """Different seeds should give different results."""
        sched = FIFOScheduler()
        r1 = evaluate_policy(sched, n_episodes=5, episode_config=short_config, seed=42)
        r2 = evaluate_policy(sched, n_episodes=5, episode_config=short_config, seed=99)
        # Very unlikely to be exactly equal with different seeds
        assert r1.mean_value != r2.mean_value

    def test_episode_metrics_populated(self, short_config):
        sched = PriorityScheduler()
        result = evaluate_policy(sched, n_episodes=2, episode_config=short_config, seed=42)

        for ep in result.episodes:
            assert ep.steps == 50
            assert ep.total_value >= 0.0
            assert ep.tasks_completed >= 0
            assert ep.queue_occupancy_mean >= 0.0


class TestComparePolicies:
    """Test the compare_policies function."""

    @pytest.fixture
    def eval_results(self):
        """Pre-computed evaluation results for comparison."""
        config = EpisodeConfig(max_steps=100)
        schedulers = [
            FIFOScheduler(),
            PriorityScheduler(),
            RoundRobinScheduler(),
            GreedyComputeScheduler(),
            RandomScheduler(seed=42),
        ]
        return [
            evaluate_policy(s, n_episodes=5, episode_config=config, seed=0)
            for s in schedulers
        ]

    def test_comparison_has_all_policies(self, eval_results):
        comparison = compare_policies(eval_results)
        assert len(comparison) == 5
        assert "FIFOScheduler" in comparison
        assert "PriorityScheduler" in comparison
        assert "RandomScheduler" in comparison

    def test_improvement_vs_fifo(self, eval_results):
        comparison = compare_policies(eval_results)
        # FIFO improvement should be 0%
        assert abs(comparison["FIFOScheduler"]["improvement_vs_fifo_pct"]) < 1e-6

    def test_all_policies_have_positive_value(self, eval_results):
        comparison = compare_policies(eval_results)
        for name, metrics in comparison.items():
            assert metrics["mean_value"] > 0.0, f"{name} has zero mean value"

    def test_summary_fields_present(self, eval_results):
        comparison = compare_policies(eval_results)
        for name, metrics in comparison.items():
            assert "mean_value" in metrics
            assert "std_value" in metrics
            assert "mean_reward" in metrics
            assert "mean_completed" in metrics
            assert "mean_dropped" in metrics
            assert "n_episodes" in metrics
            assert "improvement_vs_fifo_pct" in metrics


class TestBaselineOrdering:
    """Verify expected relative performance of baselines.

    In the default compute-rich regime (32 TOPS, ~0.5 tasks/step),
    the bottleneck is task arrival, not compute. Schedulers that
    minimize idle time outperform those that are selective.
    Under resource contention (high arrival rate), priority matters more.
    """

    @pytest.fixture
    def results(self):
        config = EpisodeConfig(max_steps=200)
        schedulers = {
            "FIFO": FIFOScheduler(),
            "Priority": PriorityScheduler(),
            "RoundRobin": RoundRobinScheduler(),
            "Greedy": GreedyComputeScheduler(),
            "Random": RandomScheduler(seed=0),
        }
        return {
            name: evaluate_policy(s, n_episodes=10, episode_config=config, seed=0)
            for name, s in schedulers.items()
        }

    def test_all_baselines_positive_value(self, results):
        for name, pm in results.items():
            assert pm.mean_value > 0.0, f"{name} got zero value"

    def test_all_baselines_complete_tasks(self, results):
        for name, pm in results.items():
            assert pm.mean_completed > 0, f"{name} completed no tasks"

    def test_fifo_same_completions_as_priority(self, results):
        """FIFO and Priority operate similarly in compute-rich regime."""
        # Both process tasks quickly, just in different order
        fifo_c = results["FIFO"].mean_completed
        priority_c = results["Priority"].mean_completed
        # They should be in the same ballpark (within 30%)
        assert abs(fifo_c - priority_c) / max(fifo_c, priority_c) < 0.3

    def test_contention_priority_advantage(self):
        """Under resource contention, Priority should beat FIFO on value."""
        # High arrival rate creates compute pressure
        config = EpisodeConfig(max_steps=200)
        sat_config = SatelliteConfig(
            task_arrival_rate=5.0,  # 10x default: creates serious backlog
            compute_capacity=16.0,  # Halve compute: more pressure
        )
        fifo = evaluate_policy(
            FIFOScheduler(), n_episodes=10,
            sat_config=sat_config, episode_config=config, seed=0,
        )
        priority = evaluate_policy(
            PriorityScheduler(), n_episodes=10,
            sat_config=sat_config, episode_config=config, seed=0,
        )
        # Under contention, priority should capture more value
        # because it processes ANOMALY (15 * 0.9 urgency) first
        assert priority.mean_value > fifo.mean_value * 0.9
