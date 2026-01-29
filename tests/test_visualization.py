"""Tests for Phase 4 visualization module."""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from satellite_edge.evaluation.metrics import (
    ExtendedEpisodeMetrics,
    ExtendedPolicyMetrics,
    ScenarioResult,
    TaskCompletionRecord,
)
from satellite_edge.evaluation.visualization import (
    get_policy_color,
    plot_policy_comparison,
    plot_scenario_heatmap,
    plot_latency_distribution,
    plot_fairness_comparison,
    plot_per_type_completion,
    create_summary_table,
    _metric_label,
)


@pytest.fixture
def sample_scenario_result():
    """Create a sample ScenarioResult for testing."""
    result = ScenarioResult(
        scenario_name="test_scenario",
        description="Test scenario for visualization tests",
    )

    # Add two policies with synthetic data
    for name, base_value in [("PolicyA", 100.0), ("PolicyB", 80.0)]:
        pm = ExtendedPolicyMetrics(policy_name=name)
        for i in range(5):
            ep = ExtendedEpisodeMetrics(
                total_value=base_value + i * 10,
                total_reward=base_value + i * 5,
                tasks_completed=50 + i,
                tasks_dropped=5,
                idle_steps=10 if name == "PolicyB" else 0,
                steps=500,
                buffer_overflow_steps=i,
                contact_time_used=300.0,
                contact_time_available=600.0,
                completions_by_type={0: 20, 1: 15, 2: 10, 3: 5 + i},
            )
            # Add task completions for latency analysis
            for j in range(20):
                ep.task_completions.append(
                    TaskCompletionRecord(
                        task_type=j % 4,
                        arrival_time=float(j),
                        completion_time=float(j + 10 + i),
                        priority=2.0 if j == 0 else 1.0,
                        value=5.0,
                    )
                )
            pm.episodes.append(ep)
        result.add_policy_result(pm)

    return result


@pytest.fixture
def sample_multi_scenario_results(sample_scenario_result):
    """Create results for multiple scenarios."""
    results = {"scenario_1": sample_scenario_result}

    # Create second scenario with different values
    result2 = ScenarioResult(
        scenario_name="scenario_2",
        description="Second test scenario",
    )
    for name, base_value in [("PolicyA", 90.0), ("PolicyB", 85.0)]:
        pm = ExtendedPolicyMetrics(policy_name=name)
        for i in range(5):
            ep = ExtendedEpisodeMetrics(
                total_value=base_value + i * 5,
                tasks_completed=40 + i,
                completions_by_type={0: 15, 1: 10, 2: 10, 3: 5},
            )
            pm.episodes.append(ep)
        result2.add_policy_result(pm)

    results["scenario_2"] = result2
    return results


class TestGetPolicyColor:
    """Tests for policy color function."""

    def test_known_policy_colors(self):
        assert get_policy_color("PPO") == "#2ecc71"
        assert get_policy_color("FIFOScheduler") == "#e74c3c"
        assert get_policy_color("ValueDensityScheduler") == "#3498db"

    def test_unknown_policy_gets_default(self):
        color = get_policy_color("UnknownPolicy")
        assert color == "#34495e"


class TestPlotPolicyComparison:
    """Tests for policy comparison bar chart."""

    def test_creates_figure(self, sample_scenario_result):
        fig = plot_policy_comparison(sample_scenario_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_metric(self, sample_scenario_result):
        fig = plot_policy_comparison(sample_scenario_result, metric="mean_completed")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_scenario_result):
        fig = plot_policy_comparison(sample_scenario_result, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotScenarioHeatmap:
    """Tests for scenario heatmap."""

    def test_creates_figure(self, sample_multi_scenario_results):
        fig = plot_scenario_heatmap(sample_multi_scenario_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_metric(self, sample_multi_scenario_results):
        fig = plot_scenario_heatmap(
            sample_multi_scenario_results, metric="mean_completed"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLatencyDistribution:
    """Tests for latency distribution violin plot."""

    def test_creates_figure(self, sample_scenario_result):
        fig = plot_latency_distribution(sample_scenario_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self, sample_scenario_result):
        fig = plot_latency_distribution(sample_scenario_result, title="Custom Title")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotFairnessComparison:
    """Tests for fairness comparison grouped bar chart."""

    def test_creates_figure(self, sample_multi_scenario_results):
        fig = plot_fairness_comparison(sample_multi_scenario_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPerTypeCompletion:
    """Tests for per-type completion stacked bar chart."""

    def test_creates_figure(self, sample_scenario_result):
        fig = plot_per_type_completion(sample_scenario_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCreateSummaryTable:
    """Tests for markdown summary table generation."""

    def test_creates_markdown(self, sample_multi_scenario_results):
        table = create_summary_table(sample_multi_scenario_results)
        assert isinstance(table, str)
        assert "# Benchmark Results Summary" in table
        assert "scenario_1" in table
        assert "scenario_2" in table

    def test_includes_policies(self, sample_multi_scenario_results):
        table = create_summary_table(sample_multi_scenario_results)
        assert "PolicyA" in table
        assert "PolicyB" in table

    def test_includes_metrics(self, sample_multi_scenario_results):
        table = create_summary_table(sample_multi_scenario_results)
        # Default metrics should be present
        assert "Value" in table or "value" in table.lower()
        assert "Completed" in table or "completed" in table.lower()

    def test_custom_metrics(self, sample_multi_scenario_results):
        table = create_summary_table(
            sample_multi_scenario_results,
            metrics=["value_iqm", "mean_completed"],
        )
        assert isinstance(table, str)


class TestMetricLabel:
    """Tests for metric label conversion."""

    def test_known_metrics(self):
        assert _metric_label("value_iqm") == "Value (IQM)"
        assert _metric_label("mean_completed") == "Tasks Completed"
        assert _metric_label("jains_fairness") == "Fairness Index"

    def test_unknown_metric_returns_itself(self):
        assert _metric_label("unknown_metric") == "unknown_metric"
