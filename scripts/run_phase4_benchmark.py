#!/usr/bin/env python3
"""CLI entry point for Phase 4 benchmark evaluation.

Runs systematic evaluation of scheduling policies across scenarios
with proper statistical methodology.

Usage:
    # Quick benchmark (CI/testing)
    uv run scripts/run_phase4_benchmark.py --quick --output results/phase4_test

    # Full benchmark
    uv run scripts/run_phase4_benchmark.py --output results/phase4 --seeds 5 --episodes 50
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    ValueDensityScheduler,
    RandomScheduler,
)
from satellite_edge.evaluation.runner import BenchmarkRunner, BenchmarkConfig
from satellite_edge.evaluation.scenarios import SCENARIOS, list_scenarios
from satellite_edge.evaluation.visualization import save_all_plots


def get_baselines() -> list:
    """Get all baseline schedulers."""
    return [
        FIFOScheduler(),
        PriorityScheduler(),
        RoundRobinScheduler(),
        GreedyComputeScheduler(),
        ValueDensityScheduler(),
        RandomScheduler(seed=42),
    ]


def try_load_ppo_agent():
    """Attempt to load a trained PPO agent if available."""
    from satellite_edge.agents.ppo_agent import SatellitePPOAgent
    from satellite_edge.environment.satellite_env import SatelliteConfig, EpisodeConfig

    # Try common model paths
    model_paths = [
        Path("models/ppo_satellite"),
        Path("models/ppo_satellite.zip"),
        Path("trained_models/ppo_satellite"),
    ]

    for path in model_paths:
        if path.exists():
            try:
                agent = SatellitePPOAgent(
                    SatelliteConfig(),
                    EpisodeConfig(),
                )
                agent.load(str(path))
                print(f"Loaded PPO agent from {path}")
                return agent
            except Exception as e:
                print(f"Warning: Could not load PPO from {path}: {e}")

    return None


class PPOWrapper:
    """Wrapper to make PPO agent compatible with BaseScheduler interface."""

    def __init__(self, agent):
        self._agent = agent

    @property
    def name(self) -> str:
        return "PPO"

    def select_action(self, obs):
        return self._agent.predict(obs, deterministic=True)

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  %(prog)s --quick --output results/test

  # Full benchmark with 5 seeds, 50 episodes each
  %(prog)s --output results/phase4 --seeds 5 --episodes 50

  # Run specific scenarios
  %(prog)s --scenarios steady_state event_surge --output results/partial
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/phase4",
        help="Output directory for results (default: results/phase4)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=50,
        help="Episodes per seed (default: 50)",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        default=5,
        help="Number of seeds for statistical validity (default: 5)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=f"Scenarios to run (default: all). Available: {list_scenarios()}",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run for CI/testing (10 episodes, 2 seeds)",
    )
    parser.add_argument(
        "--include-ppo",
        action="store_true",
        help="Attempt to include trained PPO agent",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Override for quick mode
    if args.quick:
        args.episodes = 10
        args.seeds = 2

    # Validate scenarios
    if args.scenarios:
        available = set(list_scenarios())
        invalid = set(args.scenarios) - available
        if invalid:
            print(f"Error: Unknown scenarios: {invalid}")
            print(f"Available: {list_scenarios()}")
            return 1

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        n_episodes=args.episodes,
        n_seeds=args.seeds,
        base_seed=args.base_seed,
    )

    print("=" * 60)
    print("Phase 4: Evaluation Framework Benchmark")
    print("=" * 60)
    print(f"Episodes per seed: {args.episodes}")
    print(f"Seeds: {args.seeds}")
    print(f"Scenarios: {args.scenarios or 'all'}")
    print(f"Output: {output_dir}")
    print()

    # Get policies
    policies = get_baselines()

    if args.include_ppo:
        ppo_agent = try_load_ppo_agent()
        if ppo_agent:
            policies.append(PPOWrapper(ppo_agent))
        else:
            print("Warning: No trained PPO agent found, skipping")

    print(f"Policies: {[p.name for p in policies]}")
    print()

    # Run benchmark
    runner = BenchmarkRunner(config)
    start_time = time.time()

    print("Running benchmark...")
    results = runner.run_all_scenarios(
        policies=policies,
        scenarios=args.scenarios,
    )

    elapsed = time.time() - start_time
    print(f"Benchmark completed in {elapsed:.1f}s")
    print()

    # Print summary
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    for scenario_name, result in results.items():
        print(f"\n{scenario_name} ({result.description}):")
        ranking = result.get_ranking("value_iqm")
        for i, (policy, value) in enumerate(ranking, 1):
            pm = result.policy_results[policy]
            ci_low, ci_high = pm.value_ci
            print(f"  {i}. {policy}: {value:.1f} [{ci_low:.1f}, {ci_high:.1f}]")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    runner.save_results(output_dir, results)

    if not args.skip_plots:
        print("Generating plots...")
        save_all_plots(results, output_dir)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
