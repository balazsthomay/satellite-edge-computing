"""Evaluation framework for satellite edge compute scheduling policies.

Provides rigorous benchmarking across scenarios with proper statistical methodology,
comprehensive metrics, and publication-quality visualizations.
"""

from satellite_edge.evaluation.metrics import (
    TaskCompletionRecord,
    ExtendedEpisodeMetrics,
    ExtendedPolicyMetrics,
    ScenarioResult,
    compute_iqm,
    compute_bootstrap_ci,
    compute_jains_fairness,
)
from satellite_edge.evaluation.scenarios import (
    ScenarioConfig,
    SCENARIOS,
    create_scenario_env,
)
from satellite_edge.evaluation.runner import (
    BenchmarkRunner,
)
from satellite_edge.evaluation.visualization import (
    plot_policy_comparison,
    plot_scenario_heatmap,
    plot_latency_distribution,
    create_summary_table,
)

__all__ = [
    # Metrics
    "TaskCompletionRecord",
    "ExtendedEpisodeMetrics",
    "ExtendedPolicyMetrics",
    "ScenarioResult",
    "compute_iqm",
    "compute_bootstrap_ci",
    "compute_jains_fairness",
    # Scenarios
    "ScenarioConfig",
    "SCENARIOS",
    "create_scenario_env",
    # Runner
    "BenchmarkRunner",
    # Visualization
    "plot_policy_comparison",
    "plot_scenario_heatmap",
    "plot_latency_distribution",
    "create_summary_table",
]
