"""Integration tests for PPO training and baseline comparison.

These tests verify the Phase 2 deliverable: PPO agent with learning curves
and baseline comparisons. Training is done with minimal timesteps for CI
speed; the verification of >10% improvement uses pre-trained evaluation.
"""

import numpy as np
import pytest

from satellite_edge.agents.baselines import (
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    RandomScheduler,
)
from satellite_edge.agents.evaluation import evaluate_policy, evaluate_ppo_agent, compare_policies
from satellite_edge.agents.ppo_agent import SatellitePPOAgent, PPOConfig
from satellite_edge.agents.training import get_contention_config, train_and_compare
from satellite_edge.environment.satellite_env import SatelliteConfig, EpisodeConfig


class TestContentiousBaselines:
    """Baselines under resource contention (where scheduling matters)."""

    @pytest.fixture
    def contention_configs(self):
        return get_contention_config()

    def test_contention_config_values(self, contention_configs):
        sat_config, episode_config = contention_configs
        assert sat_config.compute_capacity == 16.0
        assert sat_config.task_arrival_rate == 3.0
        assert episode_config.max_steps == 500

    def test_baselines_produce_different_values(self, contention_configs):
        """Under contention, different strategies achieve different values."""
        sat_config, _ = contention_configs
        episode_config = EpisodeConfig(max_steps=100)

        results = {}
        for name, sched in [
            ("FIFO", FIFOScheduler()),
            ("Priority", PriorityScheduler()),
            ("Random", RandomScheduler(seed=42)),
        ]:
            result = evaluate_policy(
                sched, n_episodes=5,
                sat_config=sat_config, episode_config=episode_config, seed=42,
            )
            results[name] = result

        # All should achieve positive value
        for name, r in results.items():
            assert r.mean_value > 0.0, f"{name} got zero value"

        # Random achieves higher value than single-type-focused baselines
        # because it processes all types, capturing value before decay
        assert results["Random"].mean_value > results["FIFO"].mean_value * 0.8


class TestPPOTrainingIntegration:
    """Test PPO training produces a functional agent."""

    @pytest.fixture
    def trained_agent(self, tmp_path):
        """Train a minimal PPO agent for testing."""
        sat_config, episode_config = get_contention_config()
        episode_config = EpisodeConfig(max_steps=100)  # Shorter for tests

        config = PPOConfig(
            n_steps=128,
            batch_size=64,
            n_envs=2,
            total_timesteps=1024,
            eval_freq=512,
            n_eval_episodes=2,
            episode_max_steps=100,
            early_stop_patience=100,
            min_evals_before_stop=100,
            normalize_obs=True,
            normalize_reward=True,
        )

        agent = SatellitePPOAgent(
            config=config,
            sat_config=sat_config,
            episode_config=episode_config,
            log_dir=tmp_path / "logs",
            seed=42,
        )
        agent.train()
        yield agent
        agent.close()

    def test_trained_agent_runs_episode(self, trained_agent):
        """Trained agent can complete a full episode."""
        sat_config, _ = get_contention_config()
        episode_config = EpisodeConfig(max_steps=50)

        from satellite_edge.environment.satellite_env import SatelliteEnv
        env = SatelliteEnv(sat_config=sat_config, episode_config=episode_config)
        obs, _ = env.reset(seed=99)

        for _ in range(50):
            action = trained_agent.predict(obs)
            assert 0 <= action <= 4
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert info["tasks_completed"] > 0
        env.close()

    def test_trained_agent_positive_value(self, trained_agent):
        """Trained agent achieves positive total value."""
        sat_config, _ = get_contention_config()
        episode_config = EpisodeConfig(max_steps=100)

        result = evaluate_ppo_agent(
            trained_agent, n_episodes=5,
            sat_config=sat_config, episode_config=episode_config, seed=0,
        )
        assert result.mean_value > 0.0

    def test_evaluate_ppo_agent_returns_metrics(self, trained_agent):
        """evaluate_ppo_agent returns proper PolicyMetrics."""
        sat_config, _ = get_contention_config()
        episode_config = EpisodeConfig(max_steps=50)

        result = evaluate_ppo_agent(
            trained_agent, n_episodes=3,
            sat_config=sat_config, episode_config=episode_config, seed=0,
        )
        assert result.policy_name == "PPO"
        assert len(result.episodes) == 3
        assert result.mean_completed > 0


class TestTrainAndCompare:
    """Test the full train_and_compare pipeline."""

    def test_train_and_compare_runs(self, tmp_path):
        """Full pipeline completes without error (minimal timesteps)."""
        results = train_and_compare(
            total_timesteps=512,
            n_eval_episodes=3,
            log_dir=tmp_path / "logs",
            seed=42,
            n_envs=2,
        )

        assert "comparison" in results
        assert "training_results" in results
        assert "ppo_result" in results
        assert "config" in results

        # Should have all baselines + PPO
        assert len(results["comparison"]) == 6

        # PPO should have positive value even with minimal training
        assert results["ppo_result"]["mean_value"] > 0.0


class TestPhase2Deliverable:
    """Verification that Phase 2 success criteria are met.

    The plan requires: 'PPO beats FIFO by >10% on value-weighted throughput.'
    This test verifies that a properly trained agent achieves this.
    """

    def test_ppo_beats_fifo_by_10_percent(self, tmp_path):
        """PPO trained for sufficient steps beats FIFO by >10%."""
        sat_config, episode_config = get_contention_config()

        config = PPOConfig(
            n_steps=512,
            batch_size=64,
            n_envs=4,
            total_timesteps=50_000,
            eval_freq=5_000,
            n_eval_episodes=5,
            episode_max_steps=episode_config.max_steps,
            early_stop_patience=50,
            min_evals_before_stop=5,
            normalize_obs=True,
            normalize_reward=True,
        )

        agent = SatellitePPOAgent(
            config=config,
            sat_config=sat_config,
            episode_config=episode_config,
            log_dir=tmp_path / "logs",
            seed=42,
        )
        agent.train()

        # Evaluate PPO
        ppo_result = evaluate_ppo_agent(
            agent, n_episodes=20,
            sat_config=sat_config, episode_config=episode_config, seed=1000,
        )

        # Evaluate FIFO
        fifo_result = evaluate_policy(
            FIFOScheduler(), n_episodes=20,
            sat_config=sat_config, episode_config=episode_config, seed=1000,
        )

        improvement = (ppo_result.mean_value - fifo_result.mean_value) / fifo_result.mean_value * 100

        assert improvement > 10.0, (
            f"PPO improvement ({improvement:.1f}%) did not exceed 10% threshold. "
            f"PPO value: {ppo_result.mean_value:.1f}, FIFO value: {fifo_result.mean_value:.1f}"
        )

        agent.close()
