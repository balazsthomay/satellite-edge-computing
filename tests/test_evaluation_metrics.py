"""Tests for Phase 4 evaluation metrics."""

import numpy as np
import pytest

from satellite_edge.evaluation.metrics import (
    TaskCompletionRecord,
    ExtendedEpisodeMetrics,
    ExtendedPolicyMetrics,
    ScenarioResult,
    compute_iqm,
    compute_bootstrap_ci,
    compute_jains_fairness,
)


class TestTaskCompletionRecord:
    """Tests for TaskCompletionRecord dataclass."""

    def test_basic_creation(self):
        record = TaskCompletionRecord(
            task_type=0,
            arrival_time=100.0,
            completion_time=150.0,
            priority=1.0,
            value=10.0,
        )
        assert record.task_type == 0
        assert record.arrival_time == 100.0
        assert record.completion_time == 150.0
        assert record.value == 10.0

    def test_latency_calculation(self):
        record = TaskCompletionRecord(
            task_type=1,
            arrival_time=50.0,
            completion_time=120.0,
            priority=2.0,
            value=15.0,
        )
        assert record.latency == 70.0

    def test_zero_latency(self):
        record = TaskCompletionRecord(
            task_type=0,
            arrival_time=100.0,
            completion_time=100.0,
            priority=1.0,
            value=5.0,
        )
        assert record.latency == 0.0

    def test_high_priority_flag(self):
        """High priority is defined as priority > 1.5."""
        normal = TaskCompletionRecord(0, 0.0, 10.0, priority=1.0, value=5.0)
        high = TaskCompletionRecord(0, 0.0, 10.0, priority=2.5, value=5.0)

        assert normal.priority <= 1.5
        assert high.priority > 1.5


class TestExtendedEpisodeMetrics:
    """Tests for ExtendedEpisodeMetrics dataclass."""

    def test_default_values(self):
        m = ExtendedEpisodeMetrics()
        assert m.total_value == 0.0
        assert m.tasks_completed == 0
        assert m.buffer_overflow_steps == 0
        assert m.steps == 0
        assert len(m.task_completions) == 0

    def test_contact_efficiency(self):
        m = ExtendedEpisodeMetrics(
            contact_time_used=300.0,
            contact_time_available=600.0,
        )
        assert m.contact_efficiency == 0.5

    def test_contact_efficiency_zero_available(self):
        m = ExtendedEpisodeMetrics(
            contact_time_used=0.0,
            contact_time_available=0.0,
        )
        assert m.contact_efficiency == 0.0

    def test_buffer_overflow_rate(self):
        m = ExtendedEpisodeMetrics(
            buffer_overflow_steps=50,
            steps=500,
        )
        assert m.buffer_overflow_rate == 0.1

    def test_buffer_overflow_rate_zero_steps(self):
        m = ExtendedEpisodeMetrics(steps=0)
        assert m.buffer_overflow_rate == 0.0

    def test_value_capture_efficiency(self):
        m = ExtendedEpisodeMetrics(
            total_value=80.0,
            potential_value=100.0,
        )
        assert m.value_capture_efficiency == 0.8

    def test_value_capture_efficiency_zero_potential(self):
        m = ExtendedEpisodeMetrics(potential_value=0.0)
        assert m.value_capture_efficiency == 0.0

    def test_latency_percentiles_empty(self):
        m = ExtendedEpisodeMetrics()
        p = m.latency_percentiles()
        assert p["p50"] == 0.0
        assert p["p95"] == 0.0
        assert p["p99"] == 0.0

    def test_latency_percentiles(self):
        m = ExtendedEpisodeMetrics()
        # Add 100 task completions with varying latencies
        for i in range(100):
            m.task_completions.append(
                TaskCompletionRecord(
                    task_type=i % 4,
                    arrival_time=0.0,
                    completion_time=float(i + 1),  # Latencies 1-100
                    priority=1.0,
                    value=5.0,
                )
            )

        p = m.latency_percentiles()
        assert p["p50"] == pytest.approx(50.5, rel=0.1)
        assert p["p95"] == pytest.approx(95.5, rel=0.1)
        assert p["p99"] == pytest.approx(99.5, rel=0.1)

    def test_priority_task_latencies(self):
        m = ExtendedEpisodeMetrics()
        m.task_completions = [
            TaskCompletionRecord(0, 0.0, 10.0, priority=1.0, value=5.0),  # Normal
            TaskCompletionRecord(0, 0.0, 20.0, priority=2.0, value=5.0),  # High
            TaskCompletionRecord(0, 0.0, 30.0, priority=1.5, value=5.0),  # Edge
            TaskCompletionRecord(0, 0.0, 40.0, priority=3.0, value=5.0),  # High
        ]

        high_priority_lats = m.priority_task_latencies()
        assert len(high_priority_lats) == 2
        assert 20.0 in high_priority_lats
        assert 40.0 in high_priority_lats

    def test_latencies_by_type(self):
        m = ExtendedEpisodeMetrics()
        m.task_completions = [
            TaskCompletionRecord(0, 0.0, 10.0, priority=1.0, value=5.0),
            TaskCompletionRecord(0, 0.0, 20.0, priority=1.0, value=5.0),
            TaskCompletionRecord(1, 0.0, 30.0, priority=1.0, value=5.0),
        ]

        by_type = m.latencies_by_type()
        assert len(by_type[0]) == 2
        assert by_type[0] == [10.0, 20.0]
        assert by_type[1] == [30.0]


class TestComputeIQM:
    """Tests for interquartile mean computation."""

    def test_simple_case(self):
        # 1, 2, 3, 4, 5, 6, 7, 8 -> IQM is mean of [3, 4, 5, 6] = 4.5
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        iqm = compute_iqm(values)
        assert iqm == pytest.approx(4.5, rel=0.1)

    def test_with_outliers(self):
        # IQM should be robust to outliers
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        iqm = compute_iqm(values)
        mean = np.mean(values)
        # IQM should be much lower than mean due to outlier
        assert iqm < mean

    def test_empty_list(self):
        assert compute_iqm([]) == 0.0

    def test_small_list(self):
        # Less than 4 elements falls back to mean
        values = [1.0, 2.0, 3.0]
        assert compute_iqm(values) == pytest.approx(2.0)

    def test_single_value(self):
        assert compute_iqm([5.0]) == 5.0


class TestComputeBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_basic_ci(self):
        np.random.seed(42)
        values = list(np.random.normal(100, 10, 50))

        ci_low, ci_high = compute_bootstrap_ci(
            values, confidence=0.95, n_bootstrap=1000
        )

        # CI should contain the true mean (roughly)
        assert ci_low < 100 < ci_high
        # CI should be reasonable width
        assert ci_high - ci_low < 20

    def test_single_value(self):
        ci_low, ci_high = compute_bootstrap_ci([5.0])
        assert ci_low == 5.0
        assert ci_high == 5.0

    def test_empty_list(self):
        ci_low, ci_high = compute_bootstrap_ci([])
        assert ci_low == 0.0
        assert ci_high == 0.0

    def test_reproducibility(self):
        values = list(range(20))
        rng = np.random.default_rng(12345)

        ci1 = compute_bootstrap_ci(values, rng=np.random.default_rng(12345))
        ci2 = compute_bootstrap_ci(values, rng=np.random.default_rng(12345))

        assert ci1 == ci2


class TestComputeJainsFairness:
    """Tests for Jain's fairness index computation."""

    def test_perfect_fairness(self):
        # Equal allocation = perfect fairness (J=1)
        values = [10.0, 10.0, 10.0, 10.0]
        assert compute_jains_fairness(values) == pytest.approx(1.0)

    def test_complete_unfairness(self):
        # One gets everything, others get nothing
        values = [100.0, 0.0, 0.0, 0.0]
        # J = 100^2 / (4 * 100^2) = 1/4 = 0.25
        assert compute_jains_fairness(values) == pytest.approx(0.25)

    def test_partial_fairness(self):
        # Two get resources, two don't
        values = [50.0, 50.0, 0.0, 0.0]
        # J = 100^2 / (4 * 5000) = 10000 / 20000 = 0.5
        assert compute_jains_fairness(values) == pytest.approx(0.5)

    def test_empty_list(self):
        assert compute_jains_fairness([]) == 0.0

    def test_all_zeros(self):
        # All zeros: no value distributed, so return 0 (undefined fairness)
        assert compute_jains_fairness([0.0, 0.0, 0.0, 0.0]) == 0.0


class TestExtendedPolicyMetrics:
    """Tests for ExtendedPolicyMetrics aggregation."""

    @pytest.fixture
    def policy_with_episodes(self):
        pm = ExtendedPolicyMetrics(policy_name="TestPolicy")
        for i in range(10):
            ep = ExtendedEpisodeMetrics(
                total_value=100.0 + i * 10,  # 100 to 190
                total_reward=150.0 + i * 5,
                tasks_completed=50 + i,
                tasks_dropped=5,
                buffer_overflow_steps=i,
                steps=500,
                contact_time_used=300.0,
                contact_time_available=600.0,
                potential_value=200.0,
                completions_by_type={0: 20, 1: 10, 2: 10, 3: 10 + i},
            )
            # Add task completions
            for j in range(20):
                ep.task_completions.append(
                    TaskCompletionRecord(
                        task_type=j % 4,
                        arrival_time=0.0,
                        completion_time=float(10 + j + i),
                        priority=2.0 if j == 0 else 1.0,
                        value=5.0,
                    )
                )
            pm.episodes.append(ep)
        return pm

    def test_n_episodes(self, policy_with_episodes):
        assert policy_with_episodes.n_episodes == 10

    def test_value_iqm(self, policy_with_episodes):
        iqm = policy_with_episodes.value_iqm
        # Values are 100-190, IQM should be middle range
        assert 120 < iqm < 170

    def test_value_ci(self, policy_with_episodes):
        ci_low, ci_high = policy_with_episodes.value_ci
        iqm = policy_with_episodes.value_iqm
        assert ci_low <= iqm <= ci_high

    def test_mean_value(self, policy_with_episodes):
        # Mean of 100, 110, ..., 190 = 145
        assert policy_with_episodes.mean_value == pytest.approx(145.0)

    def test_mean_completed(self, policy_with_episodes):
        assert policy_with_episodes.mean_completed == pytest.approx(54.5)

    def test_mean_buffer_overflow_rate(self, policy_with_episodes):
        # Overflows: 0 to 9, steps: 500, rates: 0.0 to 0.018
        avg_rate = policy_with_episodes.mean_buffer_overflow_rate
        assert 0.0 <= avg_rate <= 0.02

    def test_mean_contact_efficiency(self, policy_with_episodes):
        assert policy_with_episodes.mean_contact_efficiency == pytest.approx(0.5)

    def test_latency_p95_iqm(self, policy_with_episodes):
        p95_iqm = policy_with_episodes.latency_p95_iqm()
        assert p95_iqm > 0

    def test_priority_latency_iqm(self, policy_with_episodes):
        # Each episode has one high-priority task (j==0)
        iqm = policy_with_episodes.priority_latency_iqm()
        assert iqm > 0

    def test_jains_fairness(self, policy_with_episodes):
        fairness = policy_with_episodes.jains_fairness()
        # Not perfectly fair but reasonably so
        assert 0.5 < fairness <= 1.0

    def test_completion_rates_by_type(self, policy_with_episodes):
        rates = policy_with_episodes.completion_rates_by_type()
        assert len(rates) == 4
        assert rates[0] == 20.0  # All episodes have 20 type-0 completions

    def test_summary_contains_all_keys(self, policy_with_episodes):
        summary = policy_with_episodes.summary()
        required_keys = [
            "policy",
            "n_episodes",
            "value_iqm",
            "value_ci_low",
            "value_ci_high",
            "mean_value",
            "std_value",
            "mean_completed",
            "mean_dropped",
            "jains_fairness",
            "latency_p95_iqm",
            "latency_p95_ci_low",
            "latency_p95_ci_high",
            "priority_latency_iqm",
            "mean_buffer_overflow_rate",
            "mean_contact_efficiency",
            "mean_idle_fraction",
            "completion_rates_by_type",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


class TestScenarioResult:
    """Tests for ScenarioResult aggregation."""

    @pytest.fixture
    def scenario_result(self):
        result = ScenarioResult(
            scenario_name="test_scenario",
            description="Test scenario for unit tests",
        )

        # Add two policies with different performance
        for name, base_value in [("PolicyA", 100.0), ("PolicyB", 80.0)]:
            pm = ExtendedPolicyMetrics(policy_name=name)
            for i in range(5):
                ep = ExtendedEpisodeMetrics(
                    total_value=base_value + i * 5,
                    tasks_completed=50,
                    completions_by_type={0: 15, 1: 15, 2: 10, 3: 10},
                )
                pm.episodes.append(ep)
            result.add_policy_result(pm)

        return result

    def test_add_policy_result(self, scenario_result):
        assert "PolicyA" in scenario_result.policy_results
        assert "PolicyB" in scenario_result.policy_results

    def test_get_ranking(self, scenario_result):
        ranking = scenario_result.get_ranking("value_iqm")
        # PolicyA should rank higher
        assert ranking[0][0] == "PolicyA"
        assert ranking[1][0] == "PolicyB"

    def test_get_ranking_descending(self, scenario_result):
        ranking = scenario_result.get_ranking("mean_value")
        # Higher values first
        assert ranking[0][1] > ranking[1][1]

    def test_summary_table(self, scenario_result):
        table = scenario_result.summary_table()
        assert len(table) == 2
        assert all("policy" in row for row in table)
