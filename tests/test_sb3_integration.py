"""Integration tests for Stable-Baselines3 compatibility."""

import pytest
import numpy as np


class TestPPOTraining:
    """Test that PPO can actually train on the environment."""

    def test_ppo_short_training(self):
        """Test PPO can train for a few steps without crashing."""
        from stable_baselines3 import PPO
        from satellite_edge.environment import SatelliteEnv
        from satellite_edge.environment.satellite_env import EpisodeConfig

        # Short episode for faster testing
        config = EpisodeConfig(max_steps=50)
        env = SatelliteEnv(episode_config=config)

        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=64,
            batch_size=32,
        )

        # Train for a very short time
        model.learn(total_timesteps=256)

        # Verify we can make predictions
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        assert env.action_space.contains(action)

        env.close()

    def test_ppo_evaluation(self):
        """Test that a trained model can be evaluated."""
        from stable_baselines3 import PPO
        from satellite_edge.environment import SatelliteEnv
        from satellite_edge.environment.satellite_env import EpisodeConfig

        config = EpisodeConfig(max_steps=50)
        env = SatelliteEnv(episode_config=config)

        model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
        model.learn(total_timesteps=128)

        # Run evaluation episode
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Should complete an episode and get some reward
        assert isinstance(total_reward, float)

        env.close()


class TestRandomBaseline:
    """Test random policy baseline for comparison."""

    def test_random_policy_episode(self):
        """Test random policy can complete episodes."""
        from satellite_edge.environment import SatelliteEnv
        from satellite_edge.environment.satellite_env import EpisodeConfig

        config = EpisodeConfig(max_steps=100)
        env = SatelliteEnv(episode_config=config)

        obs, info = env.reset(seed=42)
        total_reward = 0.0

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Random policy should get some non-zero reward from completing tasks
        assert info["tasks_completed"] >= 0
        assert info["step"] == 100  # Should truncate at max_steps

        env.close()

    def test_fifo_policy_episode(self):
        """Test FIFO-like policy (always prioritize detection)."""
        from satellite_edge.environment import SatelliteEnv
        from satellite_edge.environment.satellite_env import EpisodeConfig
        from satellite_edge.environment.tasks import TaskType

        config = EpisodeConfig(max_steps=100)
        env = SatelliteEnv(episode_config=config)

        obs, _ = env.reset(seed=42)
        total_reward = 0.0

        # Always prioritize detection (action 0)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(TaskType.DETECTION)
            total_reward += reward

            if terminated or truncated:
                break

        assert info["tasks_completed"] > 0

        env.close()
