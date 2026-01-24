#!/usr/bin/env python3
"""Train Phase 2 PPO agent and compare against baselines.

Usage:
    uv run scripts/train_phase2.py [--timesteps N] [--n-envs N] [--output DIR]

Output (default: results/phase2/):
    model/model.zip            # Trained PPO weights
    model/vec_normalize.pkl    # Observation/reward normalization stats
    eval/evaluations.npz       # Learning curve data
    comparison.json            # Baseline comparison results
    summary.txt                # Human-readable summary
"""

import argparse
import json
import time
from pathlib import Path

import torch

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    RandomScheduler,
)
from satellite_edge.agents.evaluation import evaluate_policy, evaluate_ppo_agent, compare_policies
from satellite_edge.agents.ppo_agent import SatellitePPOAgent, PPOConfig
from satellite_edge.agents.training import get_contention_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 2 PPO agent")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Total training timesteps (default: 300k)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Parallel training environments (default: 8)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Episodes for final evaluation (default: 50)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="PyTorch device (default: auto)")
    parser.add_argument("--output", type=str, default="results/phase2",
                        help="Output directory (default: results/phase2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print training progress")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = "cpu"  # MLP is too small for GPU benefit
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Training: {args.timesteps:,} timesteps, {args.n_envs} envs")
    print(f"Output: {output_dir}")
    print()

    sat_config, episode_config = get_contention_config()

    print(f"Environment: {sat_config.compute_capacity} TOPS, "
          f"{sat_config.task_arrival_rate} tasks/step, "
          f"{episode_config.max_steps} steps/episode")
    print()

    # Configure PPO
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        eval_freq=10_000,
        n_eval_episodes=10,
        early_stop_patience=20,
        min_evals_before_stop=15,
        episode_max_steps=episode_config.max_steps,
        normalize_obs=True,
        normalize_reward=True,
        use_subprocess=False,
    )

    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    agent = SatellitePPOAgent(
        config=ppo_config,
        sat_config=sat_config,
        episode_config=episode_config,
        log_dir=output_dir,
        seed=args.seed,
    )

    t0 = time.time()
    agent.train(total_timesteps=args.timesteps)
    train_time = time.time() - t0

    print(f"\nTraining complete in {train_time:.1f}s")
    print()

    # Save model
    model_path = agent.save(output_dir / "model")
    print(f"Model saved to {model_path}")

    # Evaluate baselines
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    baselines = [
        ("FIFO", FIFOScheduler()),
        ("Priority", PriorityScheduler()),
        ("RoundRobin", RoundRobinScheduler()),
        ("GreedyCompute", GreedyComputeScheduler()),
        ("Random", RandomScheduler(seed=args.seed)),
    ]

    all_results = []
    for name, scheduler in baselines:
        result = evaluate_policy(
            scheduler, n_episodes=args.eval_episodes,
            sat_config=sat_config, episode_config=episode_config,
            seed=args.seed + 5000,
        )
        all_results.append(result)
        print(f"  {name:20s}  value={result.mean_value:8.1f} ± {result.std_value:6.1f}  "
              f"completed={result.mean_completed:5.1f}")

    # Evaluate PPO
    ppo_result = evaluate_ppo_agent(
        agent, n_episodes=args.eval_episodes,
        sat_config=sat_config, episode_config=episode_config,
        seed=args.seed + 5000,
    )
    all_results.append(ppo_result)
    print(f"  {'PPO':20s}  value={ppo_result.mean_value:8.1f} ± {ppo_result.std_value:6.1f}  "
          f"completed={ppo_result.mean_completed:5.1f}")

    # Compare
    comparison = compare_policies(all_results)

    # Summary
    fifo_val = comparison["FIFOScheduler"]["mean_value"]
    ppo_val = comparison["PPO"]["mean_value"]
    improvement = (ppo_val - fifo_val) / fifo_val * 100

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  PPO vs FIFO: +{improvement:.1f}%")
    print(f"  PPO value:   {ppo_val:.1f}")
    print(f"  FIFO value:  {fifo_val:.1f}")
    print(f"  Threshold:   >10%  {'PASS' if improvement > 10 else 'FAIL'}")

    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Phase 2 Training Results\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"Training:\n")
        f.write(f"  Timesteps: {args.timesteps:,}\n")
        f.write(f"  Envs: {args.n_envs}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Seed: {args.seed}\n")
        f.write(f"  Time: {train_time:.1f}s\n\n")
        f.write(f"Environment:\n")
        f.write(f"  Compute: {sat_config.compute_capacity} TOPS\n")
        f.write(f"  Task arrival: {sat_config.task_arrival_rate}/step\n")
        f.write(f"  Episode: {episode_config.max_steps} steps\n\n")
        f.write(f"Results ({args.eval_episodes} eval episodes):\n")
        f.write(f"  {'Policy':20s}  {'Value':>10s}  {'± Std':>8s}  {'vs FIFO':>8s}\n")
        f.write(f"  {'-' * 52}\n")
        for name, metrics in comparison.items():
            imp = metrics.get("improvement_vs_fifo_pct", 0)
            f.write(f"  {name:20s}  {metrics['mean_value']:10.1f}  "
                    f"{metrics['std_value']:8.1f}  {imp:+7.1f}%\n")
        f.write(f"\n  PPO vs FIFO: +{improvement:.1f}% ({'PASS' if improvement > 10 else 'FAIL'})\n")

    print(f"\nSaved: {comparison_path}")
    print(f"Saved: {summary_path}")

    # Save learning curve data if available
    eval_log = output_dir / "eval" / "evaluations.npz"
    if eval_log.exists():
        print(f"Learning curve: {eval_log}")

    agent.close()


if __name__ == "__main__":
    main()
