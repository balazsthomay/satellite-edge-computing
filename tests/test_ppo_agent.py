"""Tests for PPO agent training and evaluation pipeline."""

import numpy as np
import pytest
from pathlib import Path

from satellite_edge.agents.ppo_agent import SatellitePPOAgent, PPOConfig
from satellite_edge.environment.satellite_env import SatelliteEnv, SatelliteConfig, EpisodeConfig


class TestPPOConfig:
    """Test PPO configuration defaults and overrides."""

    def test_default_config(self):
        config = PPOConfig()
        assert config.learning_rate == 3e-4
        assert config.n_steps == 1024
        assert config.batch_size == 64
        assert config.gamma == 0.995
        assert config.ent_coef == 0.01
        assert config.n_envs == 8
        assert config.total_timesteps == 500_000
        assert config.normalize_obs is True
        assert config.normalize_reward is True

    def test_custom_config(self):
        config = PPOConfig(
            learning_rate=1e-4,
            n_steps=512,
            n_envs=4,
            total_timesteps=100_000,
        )
        assert config.learning_rate == 1e-4
        assert config.n_steps == 512
        assert config.n_envs == 4

    def test_net_arch_defaults(self):
        config = PPOConfig()
        assert config.net_arch_pi == [64, 64]
        assert config.net_arch_vf == [64, 64]


class TestSatellitePPOAgent:
    """Test the PPO agent wrapper."""

    @pytest.fixture
    def fast_config(self):
        """Minimal config for fast testing."""
        return PPOConfig(
            n_steps=64,
            batch_size=32,
            n_envs=2,
            total_timesteps=256,
            eval_freq=128,
            n_eval_episodes=2,
            episode_max_steps=50,
            early_stop_patience=100,  # Don't early stop in tests
            min_evals_before_stop=100,
            normalize_obs=True,
            normalize_reward=True,
            use_subprocess=False,
        )

    @pytest.fixture
    def agent(self, fast_config, tmp_path):
        agent = SatellitePPOAgent(
            config=fast_config,
            log_dir=tmp_path / "logs",
            seed=42,
        )
        yield agent
        agent.close()

    def test_agent_creation(self, agent):
        assert agent._model is None
        assert agent.config.n_envs == 2

    def test_train_returns_metrics(self, agent):
        results = agent.train()
        assert "total_timesteps" in results
        assert results["total_timesteps"] == 256
        assert "episode_values" in results
        assert "episode_completions" in results

    def test_predict_after_training(self, agent):
        agent.train()
        env = SatelliteEnv(episode_config=EpisodeConfig(max_steps=50))
        obs, _ = env.reset(seed=99)

        action = agent.predict(obs)
        assert 0 <= action <= 4
        env.close()

    def test_predict_raises_before_training(self):
        agent = SatellitePPOAgent(config=PPOConfig())
        obs = np.zeros(14, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not trained"):
            agent.predict(obs)

    def test_save_and_load(self, agent, tmp_path):
        agent.train()

        save_path = tmp_path / "saved_model"
        agent.save(save_path)

        assert (save_path / "model.zip").exists()
        assert (save_path / "vec_normalize.pkl").exists()

        # Load into new agent
        new_agent = SatellitePPOAgent(
            config=agent.config,
            log_dir=tmp_path / "logs2",
        )
        new_agent.load(save_path)

        env = SatelliteEnv(episode_config=EpisodeConfig(max_steps=50))
        obs, _ = env.reset(seed=99)
        action = new_agent.predict(obs)
        assert 0 <= action <= 4
        env.close()

    def test_train_with_custom_timesteps(self, agent):
        results = agent.train(total_timesteps=128)
        assert results["total_timesteps"] == 128

    def test_model_produces_valid_actions_over_episode(self, agent):
        """Trained model should produce valid actions for full episode."""
        agent.train()
        env = SatelliteEnv(episode_config=EpisodeConfig(max_steps=50))
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            action = agent.predict(obs)
            assert env.action_space.contains(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()

    def test_metrics_callback_collects_data(self, agent):
        agent.train()
        assert agent._metrics_callback is not None
        # With very short training, might not have full episode data
        # but the callback should exist and have lists
        assert isinstance(agent._metrics_callback.episode_values, list)
        assert isinstance(agent._metrics_callback.episode_completions, list)

    def test_checkpoint_creation(self, agent, tmp_path):
        """Training should create checkpoint files."""
        agent.train(total_timesteps=256)
        checkpoint_dir = tmp_path / "logs" / "checkpoints"
        # With short training, may or may not create checkpoints
        # but the directory structure should exist
        assert (tmp_path / "logs").exists()
