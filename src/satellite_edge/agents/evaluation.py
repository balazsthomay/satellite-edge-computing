"""Evaluation utilities for comparing scheduling policies.

Provides standardized evaluation loops, metric collection, and
comparison tools for baselines vs learned policies.
"""

from dataclasses import dataclass, field

import numpy as np

from satellite_edge.environment.satellite_env import SatelliteEnv, SatelliteConfig, EpisodeConfig
from satellite_edge.agents.baselines import BaseScheduler


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single evaluation episode."""
    total_value: float = 0.0
    total_reward: float = 0.0
    tasks_completed: int = 0
    tasks_dropped: int = 0
    buffer_overflow_steps: int = 0
    downlink_data: float = 0.0
    idle_steps: int = 0
    queue_occupancy_mean: float = 0.0
    steps: int = 0


@dataclass
class PolicyMetrics:
    """Aggregated metrics across multiple evaluation episodes."""
    policy_name: str
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_value for e in self.episodes]))

    @property
    def std_value(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.std([e.total_value for e in self.episodes]))

    @property
    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_reward for e in self.episodes]))

    @property
    def std_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.std([e.total_reward for e in self.episodes]))

    @property
    def mean_completed(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tasks_completed for e in self.episodes]))

    @property
    def mean_dropped(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tasks_dropped for e in self.episodes]))

    @property
    def mean_downlink(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.downlink_data for e in self.episodes]))

    @property
    def mean_idle_fraction(self) -> float:
        if not self.episodes:
            return 0.0
        fractions = [e.idle_steps / max(e.steps, 1) for e in self.episodes]
        return float(np.mean(fractions))

    def summary(self) -> dict[str, float]:
        """Return a summary dictionary of metrics."""
        return {
            "policy": self.policy_name,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_completed": self.mean_completed,
            "mean_dropped": self.mean_dropped,
            "mean_downlink": self.mean_downlink,
            "mean_idle_fraction": self.mean_idle_fraction,
            "n_episodes": len(self.episodes),
        }


def evaluate_policy(
    scheduler: BaseScheduler,
    n_episodes: int = 50,
    sat_config: SatelliteConfig | None = None,
    episode_config: EpisodeConfig | None = None,
    seed: int = 0,
) -> PolicyMetrics:
    """Evaluate a scheduler policy over multiple episodes.

    Args:
        scheduler: Scheduler implementing BaseScheduler interface.
        n_episodes: Number of evaluation episodes.
        sat_config: Environment configuration.
        episode_config: Episode configuration.
        seed: Base random seed (incremented per episode).

    Returns:
        PolicyMetrics with aggregated results.
    """
    sat_config = sat_config or SatelliteConfig()
    episode_config = episode_config or EpisodeConfig()

    policy_metrics = PolicyMetrics(policy_name=scheduler.name)
    num_task_types = 4  # len(TaskType)

    for ep in range(n_episodes):
        env = SatelliteEnv(
            sat_config=sat_config,
            episode_config=episode_config,
        )
        obs, info = env.reset(seed=seed + ep)
        scheduler.reset()

        ep_metrics = EpisodeMetrics()
        queue_sizes: list[float] = []

        done = False
        while not done:
            action = scheduler.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_metrics.total_reward += reward
            ep_metrics.steps += 1

            if action == num_task_types:
                ep_metrics.idle_steps += 1

            queue_sizes.append(info["queue_size"])

        ep_metrics.total_value = info["total_value"]
        ep_metrics.tasks_completed = info["tasks_completed"]
        ep_metrics.tasks_dropped = info["tasks_dropped"]
        ep_metrics.downlink_data = info["downlink_data"]
        ep_metrics.queue_occupancy_mean = float(np.mean(queue_sizes)) if queue_sizes else 0.0

        policy_metrics.episodes.append(ep_metrics)
        env.close()

    return policy_metrics


def evaluate_ppo_agent(
    agent,
    n_episodes: int = 50,
    sat_config: SatelliteConfig | None = None,
    episode_config: EpisodeConfig | None = None,
    seed: int = 0,
) -> PolicyMetrics:
    """Evaluate a trained PPO agent over multiple episodes.

    Args:
        agent: SatellitePPOAgent with a trained model.
        n_episodes: Number of evaluation episodes.
        sat_config: Environment configuration.
        episode_config: Episode configuration.
        seed: Base random seed.

    Returns:
        PolicyMetrics with aggregated results.
    """
    sat_config = sat_config or SatelliteConfig()
    episode_config = episode_config or EpisodeConfig()

    policy_metrics = PolicyMetrics(policy_name="PPO")
    num_task_types = 4

    for ep in range(n_episodes):
        env = SatelliteEnv(
            sat_config=sat_config,
            episode_config=episode_config,
        )
        obs, info = env.reset(seed=seed + ep)

        ep_metrics = EpisodeMetrics()
        queue_sizes: list[float] = []

        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_metrics.total_reward += reward
            ep_metrics.steps += 1

            if action == num_task_types:
                ep_metrics.idle_steps += 1

            queue_sizes.append(info["queue_size"])

        ep_metrics.total_value = info["total_value"]
        ep_metrics.tasks_completed = info["tasks_completed"]
        ep_metrics.tasks_dropped = info["tasks_dropped"]
        ep_metrics.downlink_data = info["downlink_data"]
        ep_metrics.queue_occupancy_mean = float(np.mean(queue_sizes)) if queue_sizes else 0.0

        policy_metrics.episodes.append(ep_metrics)
        env.close()

    return policy_metrics


def compare_policies(
    results: list[PolicyMetrics],
) -> dict[str, dict[str, float]]:
    """Compare multiple policy evaluations.

    Args:
        results: List of PolicyMetrics from evaluate_policy calls.

    Returns:
        Dictionary mapping policy names to their summary metrics,
        plus relative improvement over FIFO baseline.
    """
    comparison: dict[str, dict[str, float]] = {}

    # Find FIFO baseline for relative comparison
    fifo_value = None
    for pm in results:
        if "FIFO" in pm.policy_name:
            fifo_value = pm.mean_value
            break

    for pm in results:
        summary = pm.summary()
        if fifo_value is not None and fifo_value > 0:
            summary["improvement_vs_fifo_pct"] = (
                (pm.mean_value - fifo_value) / fifo_value * 100.0
            )
        comparison[pm.policy_name] = summary

    return comparison
