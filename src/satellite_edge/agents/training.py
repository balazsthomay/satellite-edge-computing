"""Training orchestration for PPO agent with baseline comparison.

Handles the full training lifecycle: environment setup, training,
evaluation against baselines, and result persistence.
"""

from pathlib import Path

import numpy as np

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    ValueDensityScheduler,
    RandomScheduler,
)
from satellite_edge.agents.evaluation import (
    evaluate_policy,
    evaluate_ppo_agent,
    compare_policies,
    PolicyMetrics,
)
from satellite_edge.agents.ppo_agent import SatellitePPOAgent, PPOConfig
from satellite_edge.environment.satellite_env import SatelliteConfig, EpisodeConfig


def get_contention_config() -> tuple[SatelliteConfig, EpisodeConfig]:
    """Environment config with resource contention for meaningful scheduling.

    Default config (32 TOPS, 0.5 tasks/step) is compute-rich: all tasks
    complete instantly regardless of scheduling order. For RL to learn
    anything useful, we need a regime where decisions matter.

    This config creates significant contention:
    - 12 tasks/step arrival × 3.625 avg cost = 43.5 tera-ops demand
    - 4 TOPS × 10s timestep = 40 tera-ops capacity
    - Tasks queue and decay, forcing prioritization
    - High-value ANOMALY tasks lose 59% value per step waiting
    - PPO can learn temporal patterns heuristics miss

    Under this regime, baseline ordering emerges:
    ValueDensity > RoundRobin > Priority > GreedyCompute > FIFO
    PPO typically beats best baseline by >50%.
    """
    sat_config = SatelliteConfig(
        compute_capacity=4.0,  # Constrained: creates real backlog
        buffer_capacity=256.0,
        power_capacity=300.0,
        power_per_tops=5.0,
        task_arrival_rate=12.0,  # High arrival: tasks queue and decay
        priority_event_prob=0.05,
        use_image_features=True,  # Extended obs space with image metrics
    )
    episode_config = EpisodeConfig(
        max_steps=500,  # ~1.4 hours of sim time
        timestep_duration=10.0,
    )
    return sat_config, episode_config


def train_and_compare(
    total_timesteps: int = 300_000,
    n_eval_episodes: int = 50,
    log_dir: str | Path = "./logs/phase2",
    seed: int = 42,
    n_envs: int = 8,
) -> dict:
    """Train PPO agent and compare against all baselines.

    Args:
        total_timesteps: Total training steps.
        n_eval_episodes: Episodes for final evaluation.
        log_dir: Directory for logs and models.
        seed: Random seed.
        n_envs: Parallel training environments.

    Returns:
        Dictionary with comparison results and training info.
    """
    log_dir = Path(log_dir)
    sat_config, episode_config = get_contention_config()

    # Configure PPO for contention environment
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
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        eval_freq=10_000,
        n_eval_episodes=10,
        early_stop_patience=20,
        min_evals_before_stop=15,
        episode_max_steps=episode_config.max_steps,
        normalize_obs=True,
        normalize_reward=True,
        use_subprocess=False,
    )

    # Train PPO agent
    agent = SatellitePPOAgent(
        config=ppo_config,
        sat_config=sat_config,
        episode_config=episode_config,
        log_dir=log_dir,
        seed=seed,
    )

    training_results = agent.train(total_timesteps=total_timesteps)

    # Evaluate all policies
    baselines = [
        FIFOScheduler(),
        PriorityScheduler(),
        RoundRobinScheduler(),
        GreedyComputeScheduler(),
        ValueDensityScheduler(),
        RandomScheduler(seed=seed),
    ]

    all_results: list[PolicyMetrics] = []

    for scheduler in baselines:
        result = evaluate_policy(
            scheduler,
            n_episodes=n_eval_episodes,
            sat_config=sat_config,
            episode_config=episode_config,
            seed=seed + 5000,
        )
        all_results.append(result)

    # Evaluate PPO
    ppo_result = evaluate_ppo_agent(
        agent,
        n_episodes=n_eval_episodes,
        sat_config=sat_config,
        episode_config=episode_config,
        seed=seed + 5000,
    )
    all_results.append(ppo_result)

    # Compare
    comparison = compare_policies(all_results)

    # Save model
    agent.save(log_dir / "final_model")

    agent.close()

    return {
        "comparison": comparison,
        "training_results": training_results,
        "ppo_result": ppo_result.summary(),
        "config": {
            "sat_config": {
                "compute_capacity": sat_config.compute_capacity,
                "task_arrival_rate": sat_config.task_arrival_rate,
            },
            "episode_config": {
                "max_steps": episode_config.max_steps,
            },
            "ppo_config": {
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
            },
        },
    }
