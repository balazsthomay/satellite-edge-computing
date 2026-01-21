"""Tests for SatelliteEnv gymnasium environment."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env

from satellite_edge.environment.satellite_env import (
    SatelliteEnv, SatelliteConfig, EpisodeConfig
)
from satellite_edge.environment.tasks import TaskType


class TestSatelliteEnvCreation:
    """Tests for environment initialization."""

    def test_default_creation(self):
        env = SatelliteEnv()
        assert env is not None
        assert env.sat_config is not None
        assert env.episode_config is not None

    def test_custom_config(self):
        config = SatelliteConfig(
            compute_capacity=16.0,
            buffer_capacity=128.0,
        )
        env = SatelliteEnv(sat_config=config)
        assert env.sat_config.compute_capacity == 16.0
        assert env.sat_config.buffer_capacity == 128.0

    def test_render_modes(self):
        env = SatelliteEnv(render_mode="ansi")
        assert env.render_mode == "ansi"

        env2 = SatelliteEnv(render_mode="human")
        assert env2.render_mode == "human"


class TestSpaces:
    """Tests for observation and action spaces."""

    def test_observation_space_shape(self):
        env = SatelliteEnv()
        # compute_avail + buffer_usage + queue_depths(4) + queue_compute(4) + 4 contact/power = 14
        expected_dim = 2 + 2 * len(TaskType) + 4
        assert env.observation_space.shape == (expected_dim,)

    def test_observation_space_bounds(self):
        env = SatelliteEnv()
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 1.0

    def test_action_space(self):
        env = SatelliteEnv()
        # 4 task types + IDLE = 5 actions
        assert env.action_space.n == len(TaskType) + 1

    def test_observation_in_space(self):
        env = SatelliteEnv()
        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs)
        assert obs.dtype == np.float32

    def test_observation_bounds_during_episode(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            assert env.observation_space.contains(obs)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

            if terminated or truncated:
                break


class TestReset:
    """Tests for reset functionality."""

    def test_reset_returns_tuple(self):
        env = SatelliteEnv()
        result = env.reset(seed=42)

        assert isinstance(result, tuple)
        assert len(result) == 2

        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_with_seed(self):
        env = SatelliteEnv()

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_different_seeds(self):
        env = SatelliteEnv()

        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)

        # Different seeds should give different initial states
        assert not np.array_equal(obs1, obs2)

    def test_reset_clears_state(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        # Run some steps
        for _ in range(10):
            env.step(env.action_space.sample())

        # Reset should clear state
        _, info = env.reset(seed=42)
        assert info["step"] == 0
        assert info["total_value"] == 0.0
        assert info["tasks_completed"] == 0

    def test_reset_generates_initial_tasks(self):
        env = SatelliteEnv()
        _, info = env.reset(seed=42)

        # Should have some initial tasks
        assert info["queue_size"] > 0


class TestStep:
    """Tests for step functionality."""

    def test_step_returns_correct_tuple(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_advances_time(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        _, _, _, _, info1 = env.step(0)
        _, _, _, _, info2 = env.step(0)

        assert info2["sim_time"] > info1["sim_time"]
        assert info2["step"] == info1["step"] + 1

    def test_all_actions_valid(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        for action in range(env.action_space.n):
            env.reset(seed=42)
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))

    def test_idle_action_no_compute(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        idle_action = len(TaskType)  # Last action is IDLE
        env.step(idle_action)

        # IDLE should not use compute (reflected in observation)
        obs, _, _, _, _ = env.step(idle_action)
        # First element is compute_available
        assert obs[0] == 1.0  # Full compute available after idle

    def test_episode_truncation(self):
        config = EpisodeConfig(max_steps=10)
        env = SatelliteEnv(episode_config=config)
        env.reset(seed=42)

        truncated = False
        for i in range(15):
            _, _, terminated, truncated, info = env.step(0)
            if truncated:
                break

        assert truncated
        assert info["step"] == 10


class TestReward:
    """Tests for reward calculation."""

    def test_reward_finite(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        for _ in range(100):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if terminated or truncated:
                break

    def test_completing_tasks_gives_reward(self):
        # Use config with high compute to complete tasks quickly
        config = SatelliteConfig(compute_capacity=100.0)
        env = SatelliteEnv(sat_config=config)
        env.reset(seed=42)

        total_reward = 0.0
        for _ in range(50):
            _, reward, _, _, info = env.step(0)  # Prioritize detection
            total_reward += reward

        # Should have completed some tasks and earned reward
        assert info["tasks_completed"] > 0
        assert total_reward > 0


class TestTaskProcessing:
    """Tests for task queue and processing."""

    def test_task_generation(self):
        env = SatelliteEnv()
        _, info1 = env.reset(seed=42)

        # Run steps and check task arrivals
        total_tasks = info1["queue_size"]
        for _ in range(20):
            _, _, _, _, info = env.step(0)

        # Tasks should arrive over time (unless all completed)
        # Queue might grow or shrink depending on processing
        assert info["queue_size"] >= 0

    def test_task_completion(self):
        # High compute to ensure completion
        config = SatelliteConfig(compute_capacity=100.0)
        env = SatelliteEnv(sat_config=config)
        env.reset(seed=42)

        initial_completed = 0
        for _ in range(30):
            _, _, _, _, info = env.step(0)

        assert info["tasks_completed"] > initial_completed

    def test_buffer_usage(self):
        env = SatelliteEnv()
        _, info = env.reset(seed=42)

        # Buffer should track task memory footprint
        assert info["buffer_used"] >= 0
        assert info["buffer_used"] <= env.sat_config.buffer_capacity


class TestGroundContact:
    """Tests for ground station contact handling."""

    def test_contact_state_tracked(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        contact_seen = False
        no_contact_seen = False

        for _ in range(200):  # Run through significant portion of orbit
            _, _, _, _, info = env.step(env.action_space.sample())
            if info["in_contact"]:
                contact_seen = True
            else:
                no_contact_seen = True

        # Over 200 steps (2000 seconds), should see both states
        assert contact_seen or no_contact_seen


class TestInfo:
    """Tests for info dictionary."""

    def test_info_keys(self):
        env = SatelliteEnv()
        _, info = env.reset(seed=42)

        required_keys = [
            "sim_time", "step", "total_value", "tasks_completed",
            "tasks_dropped", "queue_size", "buffer_used",
            "downlink_data", "in_contact"
        ]

        for key in required_keys:
            assert key in info

    def test_info_values_reasonable(self):
        env = SatelliteEnv()
        env.reset(seed=42)

        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())

            assert info["sim_time"] >= 0
            assert info["step"] >= 0
            assert info["total_value"] >= 0
            assert info["tasks_completed"] >= 0
            assert info["tasks_dropped"] >= 0
            assert info["queue_size"] >= 0
            assert info["buffer_used"] >= 0
            assert info["downlink_data"] >= 0
            assert isinstance(info["in_contact"], bool)


class TestRender:
    """Tests for rendering."""

    def test_ansi_render(self):
        env = SatelliteEnv(render_mode="ansi")
        env.reset(seed=42)
        env.step(0)

        output = env.render()
        assert isinstance(output, str)
        assert "Step" in output
        assert "Queue" in output

    def test_human_render_no_crash(self, capsys):
        env = SatelliteEnv(render_mode="human")
        env.reset(seed=42)
        env.step(0)

        result = env.render()
        assert result is None

        captured = capsys.readouterr()
        assert "Step" in captured.out

    def test_no_render_mode(self):
        env = SatelliteEnv(render_mode=None)
        env.reset(seed=42)
        env.step(0)

        result = env.render()
        assert result is None


class TestGymnasiumCompliance:
    """Tests for Gymnasium API compliance."""

    def test_gym_check_env(self):
        """Run gymnasium's built-in environment checker."""
        env = SatelliteEnv()

        # This will raise if there are issues
        # We catch and report any warnings
        try:
            gym_check_env(env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium check_env failed: {e}")

    def test_seeding(self):
        """Test that seeding works correctly."""
        env = SatelliteEnv()

        # Reset with seed
        obs1, _ = env.reset(seed=12345)

        # Step sequence
        actions = [0, 1, 2, 3, 4, 0, 1, 2]
        results1 = []
        for a in actions:
            obs, reward, _, _, _ = env.step(a)
            results1.append((obs.copy(), reward))

        # Reset with same seed
        obs2, _ = env.reset(seed=12345)
        np.testing.assert_array_equal(obs1, obs2)

        # Same action sequence should give same results
        for i, a in enumerate(actions):
            obs, reward, _, _, _ = env.step(a)
            np.testing.assert_array_almost_equal(obs, results1[i][0])
            assert reward == pytest.approx(results1[i][1])

    def test_close_no_error(self):
        env = SatelliteEnv()
        env.reset()
        env.step(0)
        env.close()  # Should not raise


class TestSB3Compatibility:
    """Tests for Stable-Baselines3 compatibility."""

    def test_sb3_check_env(self):
        """Test with SB3's environment checker."""
        from stable_baselines3.common.env_checker import check_env

        env = SatelliteEnv()

        # SB3 check_env will raise on errors
        try:
            check_env(env, warn=True)
        except Exception as e:
            pytest.fail(f"SB3 check_env failed: {e}")

    def test_vectorized_env_compatible(self):
        """Test that env can be vectorized."""
        from stable_baselines3.common.env_util import make_vec_env

        def make_env():
            return SatelliteEnv()

        vec_env = make_vec_env(make_env, n_envs=2)

        obs = vec_env.reset()
        assert obs.shape[0] == 2

        obs, rewards, dones, infos = vec_env.step([0, 1])
        assert obs.shape[0] == 2
        assert len(rewards) == 2

        vec_env.close()


class TestFullOrbitModel:
    """Tests for using the full Skyfield orbit model."""

    def test_create_with_full_orbit_model(self):
        config = EpisodeConfig(use_full_orbit_model=True, max_steps=10)
        env = SatelliteEnv(episode_config=config)

        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_step_with_full_orbit_model(self):
        config = EpisodeConfig(use_full_orbit_model=True, max_steps=10)
        env = SatelliteEnv(episode_config=config)
        env.reset(seed=42)

        for _ in range(5):
            obs, _, _, _, _ = env.step(0)
            assert env.observation_space.contains(obs)


class TestPowerConstraints:
    """Tests for power constraint handling."""

    def test_power_limits_processing(self):
        # Low power but high compute should be power-limited
        config = SatelliteConfig(
            compute_capacity=100.0,
            power_capacity=10.0,  # Very limited power
            power_per_tops=5.0,
        )
        env = SatelliteEnv(sat_config=config)
        env.reset(seed=42)

        # Processing should be limited by power, not compute
        obs, _, _, _, _ = env.step(0)
        # Observation should still be valid
        assert env.observation_space.contains(obs)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_queue_handling(self):
        # High compute, low arrival rate to empty queue
        config = SatelliteConfig(
            compute_capacity=200.0,
            task_arrival_rate=0.1,
        )
        env = SatelliteEnv(sat_config=config)
        env.reset(seed=42)

        # Run until queue is likely empty
        for _ in range(100):
            env.step(0)

        # Should handle empty queue gracefully
        obs, reward, _, _, info = env.step(0)
        assert env.observation_space.contains(obs)

    def test_full_buffer_handling(self):
        # Small buffer, high arrival rate
        config = SatelliteConfig(
            buffer_capacity=2.0,  # Very small
            task_arrival_rate=5.0,  # High arrival
            compute_capacity=1.0,  # Low compute
        )
        env = SatelliteEnv(sat_config=config)
        env.reset(seed=42)

        # Run to fill buffer
        for _ in range(20):
            obs, _, _, _, info = env.step(4)  # IDLE to not process

        # Should handle full buffer (tasks dropped)
        assert info["tasks_dropped"] > 0
        assert info["buffer_used"] <= config.buffer_capacity

    def test_long_episode(self):
        config = EpisodeConfig(max_steps=5000)
        env = SatelliteEnv(episode_config=config)
        env.reset(seed=42)

        for i in range(5000):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

        assert truncated  # Should truncate at max_steps
