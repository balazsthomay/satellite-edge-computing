"""Tests for Phase 4 scenario definitions."""

import pytest

from satellite_edge.evaluation.scenarios import (
    ScenarioConfig,
    SCENARIOS,
    create_scenario_env,
    get_scenario,
    list_scenarios,
)
from satellite_edge.environment.satellite_env import SatelliteEnv


class TestScenarioConfig:
    """Tests for ScenarioConfig dataclass."""

    def test_basic_creation(self):
        from satellite_edge.environment.satellite_env import (
            SatelliteConfig,
            EpisodeConfig,
        )

        config = ScenarioConfig(
            name="test",
            description="Test scenario",
            sat_config=SatelliteConfig(),
            episode_config=EpisodeConfig(),
        )
        assert config.name == "test"
        assert config.description == "Test scenario"

    def test_create_env(self):
        config = SCENARIOS["steady_state"]
        env = config.create_env(seed=42)
        assert isinstance(env, SatelliteEnv)
        env.close()


class TestSCENARIOS:
    """Tests for pre-defined scenarios."""

    def test_all_scenarios_exist(self):
        expected = ["steady_state", "event_surge", "degraded_compute", "long_blackout"]
        for name in expected:
            assert name in SCENARIOS

    def test_steady_state_config(self):
        config = SCENARIOS["steady_state"]
        assert config.name == "steady_state"
        assert config.sat_config.compute_capacity == 4.0
        assert config.sat_config.task_arrival_rate == 12.0
        assert config.sat_config.priority_event_prob == 0.05

    def test_event_surge_config(self):
        config = SCENARIOS["event_surge"]
        assert config.name == "event_surge"
        # Higher arrival rate and priority probability
        assert config.sat_config.task_arrival_rate == 25.0
        assert config.sat_config.priority_event_prob == 0.30

    def test_degraded_compute_config(self):
        config = SCENARIOS["degraded_compute"]
        assert config.name == "degraded_compute"
        # Lower compute capacity
        assert config.sat_config.compute_capacity == 2.0
        assert config.sat_config.task_arrival_rate == 12.0

    def test_long_blackout_config(self):
        config = SCENARIOS["long_blackout"]
        assert config.name == "long_blackout"
        # Smaller buffer and modified orbit config
        assert config.sat_config.buffer_capacity == 128.0
        assert config.orbit_config is not None
        assert config.orbit_config["contact_fraction"] == 0.05


class TestCreateScenarioEnv:
    """Tests for create_scenario_env function."""

    def test_creates_valid_env(self):
        config = SCENARIOS["steady_state"]
        env = create_scenario_env(config, seed=42)

        assert isinstance(env, SatelliteEnv)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_env_uses_scenario_config(self):
        config = SCENARIOS["degraded_compute"]
        env = create_scenario_env(config, seed=42)

        assert env.sat_config.compute_capacity == 2.0
        env.close()

    def test_env_runs_episode(self):
        config = SCENARIOS["event_surge"]
        env = create_scenario_env(config, seed=42)
        env.reset(seed=42)

        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(0)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

        env.close()

    def test_all_scenarios_create_valid_envs(self):
        for name, config in SCENARIOS.items():
            env = create_scenario_env(config, seed=42)
            obs, _ = env.reset(seed=42)
            assert env.observation_space.contains(obs), f"{name} has invalid obs"
            env.close()

    def test_orbit_config_applied(self):
        config = SCENARIOS["long_blackout"]
        env = create_scenario_env(config, seed=42)

        # Check that orbit model has modified contact fraction
        assert hasattr(env, "_orbit")
        if hasattr(env._orbit, "contact_fraction"):
            assert env._orbit.contact_fraction == 0.05

        env.close()


class TestGetScenario:
    """Tests for get_scenario helper function."""

    def test_get_existing_scenario(self):
        config = get_scenario("steady_state")
        assert config.name == "steady_state"

    def test_get_unknown_scenario(self):
        with pytest.raises(KeyError) as exc_info:
            get_scenario("nonexistent_scenario")
        assert "nonexistent_scenario" in str(exc_info.value)


class TestListScenarios:
    """Tests for list_scenarios helper function."""

    def test_returns_all_scenario_names(self):
        names = list_scenarios()
        assert "steady_state" in names
        assert "event_surge" in names
        assert "degraded_compute" in names
        assert "long_blackout" in names

    def test_returns_list(self):
        names = list_scenarios()
        assert isinstance(names, list)


class TestScenarioContention:
    """Tests that verify scenario contention characteristics."""

    def test_steady_state_moderate_contention(self):
        """Steady state should have moderate resource contention."""
        config = SCENARIOS["steady_state"]
        env = create_scenario_env(config, seed=42)
        env.reset(seed=42)

        # Run for a while and check queue doesn't explode
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Queue should be manageable (not infinite growth)
        assert info["queue_size"] < 100, "Queue should not grow unboundedly"
        assert info["tasks_completed"] > 0, "Should complete some tasks"

        env.close()

    def test_event_surge_high_contention(self):
        """Event surge should create high resource pressure."""
        config = SCENARIOS["event_surge"]
        env = create_scenario_env(config, seed=42)
        env.reset(seed=42)

        # Run and check for high task arrival/completion ratio
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Should see tasks dropped or large queue under surge
        total_tasks_seen = (
            info["tasks_completed"] + info["tasks_dropped"] + info["queue_size"]
        )
        assert total_tasks_seen > info["tasks_completed"], (
            "High arrival should exceed processing"
        )

        env.close()

    def test_degraded_compute_creates_backlog(self):
        """Degraded compute should create processing backlog."""
        config = SCENARIOS["degraded_compute"]
        env = create_scenario_env(config, seed=42)
        env.reset(seed=42)

        # With half the compute, queue should build up
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Should have noticeable queue backlog
        assert info["queue_size"] > 5, "Queue should build up with degraded compute"

        env.close()

    def test_long_blackout_buffer_pressure(self):
        """Long blackout should create buffer pressure."""
        config = SCENARIOS["long_blackout"]
        env = create_scenario_env(config, seed=42)
        env.reset(seed=42)

        max_buffer = 0
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(0)
            max_buffer = max(max_buffer, info["buffer_used"])
            if terminated or truncated:
                break

        # Buffer should get reasonably utilized
        assert max_buffer > 10, "Buffer should be utilized under blackout"

        env.close()


class TestScenarioDeterminism:
    """Tests for scenario reproducibility."""

    def test_same_seed_same_results(self):
        """Same seed should produce identical episode."""
        config = SCENARIOS["steady_state"]

        env1 = create_scenario_env(config, seed=42)
        obs1, _ = env1.reset(seed=42)
        rewards1 = []
        for _ in range(50):
            _, reward, _, _, _ = env1.step(0)
            rewards1.append(reward)
        env1.close()

        env2 = create_scenario_env(config, seed=42)
        obs2, _ = env2.reset(seed=42)
        rewards2 = []
        for _ in range(50):
            _, reward, _, _, _ = env2.step(0)
            rewards2.append(reward)
        env2.close()

        assert obs1.tolist() == obs2.tolist()
        assert rewards1 == rewards2

    def test_different_seeds_different_results(self):
        """Different seeds should produce different episodes."""
        config = SCENARIOS["steady_state"]

        env1 = create_scenario_env(config, seed=42)
        env1.reset(seed=42)
        total1 = 0
        for _ in range(50):
            _, reward, _, _, _ = env1.step(0)
            total1 += reward
        env1.close()

        env2 = create_scenario_env(config, seed=123)
        env2.reset(seed=123)
        total2 = 0
        for _ in range(50):
            _, reward, _, _, _ = env2.step(0)
            total2 += reward
        env2.close()

        # Highly unlikely to be exactly equal with different seeds
        assert total1 != total2
