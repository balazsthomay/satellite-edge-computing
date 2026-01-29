"""Visualization utilities for benchmark results.

Provides publication-quality plots for policy comparison.
"""

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from satellite_edge.evaluation.metrics import (
    ExtendedPolicyMetrics,
    ScenarioResult,
)


# Color scheme for consistent policy colors
POLICY_COLORS = {
    "PPO": "#2ecc71",  # Green
    "ValueDensityScheduler": "#3498db",  # Blue
    "PriorityScheduler": "#9b59b6",  # Purple
    "FIFOScheduler": "#e74c3c",  # Red
    "GreedyComputeScheduler": "#f39c12",  # Orange
    "RoundRobinScheduler": "#1abc9c",  # Teal
    "RandomScheduler": "#95a5a6",  # Gray
}

DEFAULT_COLOR = "#34495e"  # Dark gray for unknown policies


def get_policy_color(name: str) -> str:
    """Get consistent color for a policy."""
    return POLICY_COLORS.get(name, DEFAULT_COLOR)


def plot_policy_comparison(
    result: ScenarioResult,
    metric: str = "value_iqm",
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
) -> plt.Figure:
    """Create bar chart comparing policies on a single metric.

    Args:
        result: ScenarioResult with policy evaluations.
        metric: Metric to plot (from summary()).
        figsize: Figure size.
        title: Plot title (auto-generated if None).

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    policies = list(result.policy_results.keys())
    summaries = [result.policy_results[p].summary() for p in policies]

    values = [s[metric] for s in summaries]
    colors = [get_policy_color(p) for p in policies]

    # Get CIs if available
    ci_metric = metric.replace("_iqm", "_ci")
    has_ci = f"{ci_metric}_low" in summaries[0]

    x = np.arange(len(policies))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    if has_ci:
        ci_low = [s[f"{ci_metric}_low"] for s in summaries]
        ci_high = [s[f"{ci_metric}_high"] for s in summaries]
        errors = [
            [v - lo for v, lo in zip(values, ci_low)],
            [hi - v for v, hi in zip(values, ci_high)],
        ]
        ax.errorbar(
            x, values, yerr=errors, fmt="none", color="black", capsize=4, capthick=1.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel(_metric_label(metric))

    if title is None:
        title = f"{result.scenario_name}: {_metric_label(metric)}"
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_scenario_heatmap(
    results: dict[str, ScenarioResult],
    metric: str = "value_iqm",
    figsize: tuple[float, float] = (12, 8),
    title: str = "Policy Performance Across Scenarios",
) -> plt.Figure:
    """Create heatmap of policy × scenario performance.

    Args:
        results: Dict mapping scenario names to ScenarioResults.
        metric: Metric to visualize.
        figsize: Figure size.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    scenarios = list(results.keys())
    first_result = results[scenarios[0]]
    policies = list(first_result.policy_results.keys())

    # Build data matrix
    data = np.zeros((len(scenarios), len(policies)))
    for i, scenario in enumerate(scenarios):
        for j, policy in enumerate(policies):
            if policy in results[scenario].policy_results:
                summary = results[scenario].policy_results[policy].summary()
                data[i, j] = summary.get(metric, 0)

    # Normalize by row (scenario) for fair comparison
    row_max = data.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1  # Avoid division by zero
    normalized = data / row_max

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(policies)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_yticklabels(scenarios)

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(policies)):
            text_color = "white" if normalized[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, label="Relative Performance")

    plt.tight_layout()
    return fig


def plot_latency_distribution(
    result: ScenarioResult,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> plt.Figure:
    """Create violin plot of latency distributions by policy.

    Args:
        result: ScenarioResult with policy evaluations.
        figsize: Figure size.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    policies = list(result.policy_results.keys())
    all_latencies = []
    positions = []

    for i, policy in enumerate(policies):
        pm = result.policy_results[policy]
        latencies = []
        for ep in pm.episodes:
            latencies.extend([tc.latency for tc in ep.task_completions])

        if latencies:
            all_latencies.append(latencies)
            positions.append(i)

    if all_latencies:
        parts = ax.violinplot(
            all_latencies,
            positions=positions,
            showmeans=True,
            showmedians=True,
        )

        # Color the violins
        for i, (pc, pos) in enumerate(zip(parts["bodies"], positions)):
            color = get_policy_color(policies[pos])
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Task Latency (seconds)")

    if title is None:
        title = f"{result.scenario_name}: Task Latency Distribution"
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_fairness_comparison(
    results: dict[str, ScenarioResult],
    figsize: tuple[float, float] = (10, 6),
    title: str = "Jain's Fairness Index by Scenario",
) -> plt.Figure:
    """Create grouped bar chart of Jain's fairness index.

    Args:
        results: Dict mapping scenario names to ScenarioResults.
        figsize: Figure size.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    scenarios = list(results.keys())
    first_result = results[scenarios[0]]
    policies = list(first_result.policy_results.keys())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(scenarios))
    width = 0.8 / len(policies)

    for i, policy in enumerate(policies):
        fairness_values = []
        for scenario in scenarios:
            if policy in results[scenario].policy_results:
                pm = results[scenario].policy_results[policy]
                fairness_values.append(pm.jains_fairness())
            else:
                fairness_values.append(0)

        offset = (i - len(policies) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            fairness_values,
            width,
            label=policy,
            color=get_policy_color(policy),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.05)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_per_type_completion(
    result: ScenarioResult,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> plt.Figure:
    """Create stacked bar chart of completions by task type.

    Args:
        result: ScenarioResult with policy evaluations.
        figsize: Figure size.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    from satellite_edge.environment.tasks import TaskType

    fig, ax = plt.subplots(figsize=figsize)

    policies = list(result.policy_results.keys())
    task_types = list(TaskType)
    type_colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
    ]  # Blue, Red, Green, Orange

    x = np.arange(len(policies))
    width = 0.6
    bottoms = np.zeros(len(policies))

    for i, task_type in enumerate(task_types):
        type_counts = []
        for policy in policies:
            pm = result.policy_results[policy]
            rates = pm.completion_rates_by_type()
            type_counts.append(rates.get(int(task_type), 0))

        ax.bar(
            x,
            type_counts,
            width,
            bottom=bottoms,
            label=task_type.name,
            color=type_colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        bottoms += np.array(type_counts)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Mean Task Completions per Episode")
    ax.legend(title="Task Type", loc="upper left", bbox_to_anchor=(1, 1))

    if title is None:
        title = f"{result.scenario_name}: Task Completions by Type"
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def create_summary_table(
    results: dict[str, ScenarioResult],
    metrics: list[str] | None = None,
) -> str:
    """Generate markdown table summarizing benchmark results.

    Args:
        results: Dict mapping scenario names to ScenarioResults.
        metrics: Metrics to include (default: key metrics).

    Returns:
        Markdown-formatted table string.
    """
    if metrics is None:
        metrics = [
            "value_iqm",
            "mean_completed",
            "latency_p95_iqm",
            "jains_fairness",
            "mean_buffer_overflow_rate",
        ]

    lines = ["# Benchmark Results Summary\n"]

    for scenario_name, result in results.items():
        lines.append(f"## {scenario_name}\n")
        lines.append(f"*{result.description}*\n")

        # Build header
        header = "| Policy |"
        separator = "|--------|"
        for m in metrics:
            header += f" {_metric_label(m)} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Build rows, sorted by value_iqm
        ranking = result.get_ranking("value_iqm")
        for policy_name, _ in ranking:
            pm = result.policy_results[policy_name]
            summary = pm.summary()

            row = f"| {policy_name} |"
            for m in metrics:
                val = summary.get(m, 0)
                if isinstance(val, float):
                    # Format CIs if available
                    ci_key = m.replace("_iqm", "_ci")
                    if f"{ci_key}_low" in summary and "_iqm" in m:
                        ci_low = summary[f"{ci_key}_low"]
                        ci_high = summary[f"{ci_key}_high"]
                        row += f" {val:.1f} [{ci_low:.1f}, {ci_high:.1f}] |"
                    else:
                        row += f" {val:.2f} |"
                else:
                    row += f" {val} |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def save_all_plots(
    results: dict[str, ScenarioResult],
    output_dir: str | Path,
) -> None:
    """Save all benchmark plots to disk.

    Args:
        results: Benchmark results.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-scenario plots
    for scenario_name, result in results.items():
        # Value comparison
        fig = plot_policy_comparison(result, "value_iqm")
        fig.savefig(
            plots_dir / f"{scenario_name}_value.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # Latency distribution
        fig = plot_latency_distribution(result)
        fig.savefig(
            plots_dir / f"{scenario_name}_latency.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # Per-type completions
        fig = plot_per_type_completion(result)
        fig.savefig(
            plots_dir / f"{scenario_name}_completions.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Cross-scenario plots
    fig = plot_scenario_heatmap(results)
    fig.savefig(plots_dir / "scenario_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_fairness_comparison(results)
    fig.savefig(plots_dir / "fairness_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary table
    summary_md = create_summary_table(results)
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)


def _metric_label(metric: str) -> str:
    """Get human-readable label for a metric."""
    labels = {
        "value_iqm": "Value (IQM)",
        "mean_value": "Mean Value",
        "mean_completed": "Tasks Completed",
        "mean_dropped": "Tasks Dropped",
        "latency_p95_iqm": "P95 Latency (s)",
        "priority_latency_iqm": "Priority Latency (s)",
        "jains_fairness": "Fairness Index",
        "mean_buffer_overflow_rate": "Overflow Rate",
        "mean_contact_efficiency": "Contact Efficiency",
        "mean_idle_fraction": "Idle Fraction",
    }
    return labels.get(metric, metric)
