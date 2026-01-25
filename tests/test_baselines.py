"""Tests for baseline scheduling policies."""

import numpy as np
import pytest

from satellite_edge.agents.baselines import (
    BaseScheduler,
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    ValueDensityScheduler,
    RandomScheduler,
)
from satellite_edge.environment.satellite_env import SatelliteEnv, SatelliteConfig, EpisodeConfig
from satellite_edge.environment.tasks import TaskType, TASK_SPECS


# ─── Observation fixtures ───────────────────────────────────────────────

def make_obs(
    queue_depths: list[float] | None = None,
    queue_compute: list[float] | None = None,
    compute_avail: float = 1.0,
    buffer_usage: float = 0.2,
    power_avail: float = 1.0,
) -> np.ndarray:
    """Create a mock observation vector for testing schedulers."""
    if queue_depths is None:
        queue_depths = [0.1, 0.05, 0.1, 0.1]  # normalized by 50

    if queue_compute is None:
        queue_compute = [d * 0.5 for d in queue_depths]  # arbitrary default

    # obs[0:2] = compute_avail, buffer_usage
    # obs[2:6] = queue_depths
    # obs[6:10] = queue_compute
    # obs[10:14] = ground_contact, time_to_contact, contact_duration, power_avail
    obs = np.array(
        [compute_avail, buffer_usage]
        + queue_depths
        + queue_compute
        + [0.0, 0.5, 0.0, power_avail],
        dtype=np.float32,
    )
    return obs


# ─── Never-idle property tests ─────────────────────────────────────────

class TestNeverIdleProperty:
    """All deterministic baselines must never return IDLE action."""

    @pytest.mark.parametrize("scheduler_cls", [
        FIFOScheduler,
        PriorityScheduler,
        RoundRobinScheduler,
        GreedyComputeScheduler,
        ValueDensityScheduler,
    ])
    def test_never_idles_on_empty_queues(self, scheduler_cls):
        """Baselines should not idle even when queues appear empty."""
        sched = scheduler_cls()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        action = sched.select_action(obs)
        assert action != len(TaskType), f"{scheduler_cls.__name__} idled on empty queues"

    @pytest.mark.parametrize("scheduler_cls", [
        FIFOScheduler,
        PriorityScheduler,
        RoundRobinScheduler,
        GreedyComputeScheduler,
        ValueDensityScheduler,
    ])
    def test_never_idles_across_many_steps(self, scheduler_cls):
        """Verify never-idle over many consecutive calls."""
        sched = scheduler_cls()
        idle_action = len(TaskType)

        for _ in range(100):
            # Vary queue states
            depths = np.random.rand(4) * 0.2
            obs = make_obs(queue_depths=depths.tolist())
            action = sched.select_action(obs)
            assert action != idle_action, f"{scheduler_cls.__name__} idled"


# ─── FIFOScheduler tests ────────────────────────────────────────────────

class TestFIFOScheduler:
    def test_selects_deepest_queue(self):
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.1, 0.3, 0.05, 0.2])
        assert sched.select_action(obs) == 1  # ANOMALY has deepest queue

    def test_defaults_to_detection_on_empty_queues(self):
        """Empty queues should default to DETECTION (highest arrival weight)."""
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == int(TaskType.DETECTION)

    def test_handles_tie_deterministically(self):
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.2, 0.2, 0.2, 0.2])
        action = sched.select_action(obs)
        assert action == 0  # argmax returns first occurrence

    def test_single_type_with_tasks(self):
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.3, 0.0])
        assert sched.select_action(obs) == 2  # CLOUD_MASK

    def test_name(self):
        assert FIFOScheduler().name == "FIFOScheduler"


# ─── PriorityScheduler tests ───────────────────────────────────────────

class TestPriorityScheduler:
    def test_selects_by_weighted_priority(self):
        """Priority = queue_depth × value_rate."""
        sched = PriorityScheduler()
        # Equal depths: ANOMALY wins (13.5 > 8.0 > 2.5 > 0.9)
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_depth_can_override_value_rate(self):
        """High depth on low-value type can win."""
        sched = PriorityScheduler()
        # CLOUD_MASK at 0.5 depth: 0.5 * 0.9 = 0.45
        # ANOMALY at 0.01 depth: 0.01 * 13.5 = 0.135
        obs = make_obs(queue_depths=[0.0, 0.01, 0.5, 0.0])
        assert sched.select_action(obs) == int(TaskType.CLOUD_MASK)

    def test_defaults_to_anomaly_on_empty(self):
        """Empty queues default to ANOMALY (highest value rate)."""
        sched = PriorityScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_value_rates_correct(self):
        """Verify pre-computed value rates match task specs."""
        sched = PriorityScheduler()
        # DETECTION: 10 * 0.8 = 8.0
        # ANOMALY: 15 * 0.9 = 13.5
        # CLOUD_MASK: 3 * 0.3 = 0.9
        # COMPRESSION: 5 * 0.5 = 2.5
        expected = np.array([8.0, 13.5, 0.9, 2.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(sched._value_rates, expected)


# ─── RoundRobinScheduler tests ─────────────────────────────────────────

class TestRoundRobinScheduler:
    def test_cycles_unconditionally(self):
        """Cycles through all types regardless of queue state."""
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])

        actions = [sched.select_action(obs) for _ in range(8)]
        assert actions == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_cycles_even_with_empty_queues(self):
        """Never skips types even if queues appear empty."""
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])

        actions = [sched.select_action(obs) for _ in range(4)]
        assert actions == [0, 1, 2, 3]

    def test_reset_restarts_cycle(self):
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])

        sched.select_action(obs)  # 0
        sched.select_action(obs)  # 1
        sched.reset()
        assert sched.select_action(obs) == 0  # Back to start


# ─── GreedyComputeScheduler tests ──────────────────────────────────────

class TestGreedyComputeScheduler:
    def test_selects_highest_compute_backlog(self):
        sched = GreedyComputeScheduler()
        obs = make_obs(
            queue_depths=[0.1, 0.1, 0.1, 0.1],
            queue_compute=[0.1, 0.3, 0.05, 0.2],  # ANOMALY has most compute
        )
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_defaults_to_detection_on_empty(self):
        """Empty compute defaults to DETECTION (most compute-heavy)."""
        sched = GreedyComputeScheduler()
        obs = make_obs(
            queue_depths=[0.0, 0.0, 0.0, 0.0],
            queue_compute=[0.0, 0.0, 0.0, 0.0],
        )
        assert sched.select_action(obs) == int(TaskType.DETECTION)

    def test_uses_queue_compute_not_depth(self):
        """Should use obs[6:10] (compute) not obs[2:6] (depth)."""
        sched = GreedyComputeScheduler()
        obs = make_obs(
            queue_depths=[0.5, 0.1, 0.1, 0.1],  # DETECTION has most depth
            queue_compute=[0.1, 0.1, 0.5, 0.1],  # CLOUD_MASK has most compute
        )
        assert sched.select_action(obs) == int(TaskType.CLOUD_MASK)


# ─── ValueDensityScheduler tests ───────────────────────────────────────

class TestValueDensityScheduler:
    def test_selects_highest_density_with_tasks(self):
        """With equal depths, picks ANOMALY (highest density)."""
        sched = ValueDensityScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_skips_empty_high_density(self):
        """Skips types without tasks even if high density."""
        sched = ValueDensityScheduler()
        obs = make_obs(queue_depths=[0.1, 0.0, 0.1, 0.1])  # ANOMALY empty
        # Next highest density is DETECTION (1.33)
        assert sched.select_action(obs) == int(TaskType.DETECTION)

    def test_defaults_to_anomaly_on_empty(self):
        sched = ValueDensityScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_density_values_correct(self):
        """Verify pre-computed density values match task specs."""
        sched = ValueDensityScheduler()
        # DETECTION: (10 * 0.8) / 6.0 = 1.333
        # ANOMALY: (15 * 0.9) / 3.0 = 4.5
        # CLOUD_MASK: (3 * 0.3) / 1.5 = 0.6
        # COMPRESSION: (5 * 0.5) / 4.0 = 0.625
        expected = np.array([1.333333, 4.5, 0.6, 0.625], dtype=np.float32)
        np.testing.assert_array_almost_equal(sched._density, expected, decimal=4)

    def test_priority_order_correct(self):
        """Verify density ordering: ANOMALY > DETECTION > COMPRESSION > CLOUD_MASK."""
        sched = ValueDensityScheduler()
        expected_order = [
            int(TaskType.ANOMALY),     # 4.5
            int(TaskType.DETECTION),   # 1.33
            int(TaskType.COMPRESSION), # 0.625
            int(TaskType.CLOUD_MASK),  # 0.6
        ]
        assert sched._priority_order == expected_order


# ─── RandomScheduler tests ──────────────────────────────────────────────

class TestRandomScheduler:
    def test_actions_in_valid_range(self):
        sched = RandomScheduler(seed=42)
        obs = make_obs()
        actions = [sched.select_action(obs) for _ in range(100)]
        assert all(0 <= a <= len(TaskType) for a in actions)

    def test_produces_all_actions_including_idle(self):
        sched = RandomScheduler(seed=42)
        obs = make_obs()
        actions = set(sched.select_action(obs) for _ in range(1000))
        assert actions == {0, 1, 2, 3, 4}

    def test_deterministic_with_seed(self):
        sched1 = RandomScheduler(seed=123)
        sched2 = RandomScheduler(seed=123)
        obs = make_obs()
        actions1 = [sched1.select_action(obs) for _ in range(20)]
        actions2 = [sched2.select_action(obs) for _ in range(20)]
        assert actions1 == actions2


# ─── Integration: baselines on real environment ─────────────────────────

class TestBaselineIntegration:
    """Run each baseline on the actual SatelliteEnv to verify compatibility."""

    @pytest.fixture
    def env(self):
        config = EpisodeConfig(max_steps=100)
        env = SatelliteEnv(episode_config=config)
        yield env
        env.close()

    @pytest.mark.parametrize("scheduler_cls", [
        FIFOScheduler,
        PriorityScheduler,
        RoundRobinScheduler,
        GreedyComputeScheduler,
        ValueDensityScheduler,
    ])
    def test_baseline_completes_episode(self, env, scheduler_cls):
        """Each baseline should complete a full episode without error."""
        scheduler = scheduler_cls()
        obs, info = env.reset(seed=42)
        scheduler.reset()

        total_reward = 0.0
        for _ in range(100):
            action = scheduler.select_action(obs)
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        assert info["step"] == 100
        assert info["tasks_completed"] >= 0

    @pytest.mark.parametrize("scheduler_cls", [
        FIFOScheduler,
        PriorityScheduler,
        RoundRobinScheduler,
        GreedyComputeScheduler,
        ValueDensityScheduler,
    ])
    def test_baseline_gets_positive_value(self, env, scheduler_cls):
        """Non-random baselines should achieve positive task value."""
        scheduler = scheduler_cls()
        obs, _ = env.reset(seed=42)
        scheduler.reset()

        for _ in range(100):
            action = scheduler.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert info["total_value"] > 0.0

    @pytest.mark.parametrize("scheduler_cls", [
        FIFOScheduler,
        PriorityScheduler,
        RoundRobinScheduler,
        GreedyComputeScheduler,
        ValueDensityScheduler,
    ])
    def test_baseline_zero_idle_fraction(self, scheduler_cls):
        """Deterministic baselines should never idle."""
        scheduler = scheduler_cls()
        config = EpisodeConfig(max_steps=200)
        env = SatelliteEnv(episode_config=config)
        obs, _ = env.reset(seed=42)
        scheduler.reset()

        idle_steps = 0
        for _ in range(200):
            action = scheduler.select_action(obs)
            if action == len(TaskType):
                idle_steps += 1
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert idle_steps == 0, f"{scheduler.name} idled {idle_steps} times"
        env.close()

    def test_random_baseline_episode(self, env):
        scheduler = RandomScheduler(seed=42)
        obs, _ = env.reset(seed=42)

        for _ in range(100):
            action = scheduler.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert info["step"] == 100

    def test_baselines_complete_tasks(self):
        """All baselines should complete tasks and achieve positive value."""
        config = EpisodeConfig(max_steps=200)

        for scheduler_cls in [
            FIFOScheduler,
            PriorityScheduler,
            RoundRobinScheduler,
            GreedyComputeScheduler,
            ValueDensityScheduler,
        ]:
            scheduler = scheduler_cls()
            env = SatelliteEnv(episode_config=config)
            obs, _ = env.reset(seed=42)
            scheduler.reset()

            for _ in range(200):
                action = scheduler.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            assert info["total_value"] > 0.0, f"{scheduler.name} got zero value"
            assert info["tasks_completed"] > 0, f"{scheduler.name} completed no tasks"
            env.close()

    def test_priority_selects_high_value_types(self):
        """Priority scheduler should preferentially select ANOMALY when available."""
        config = EpisodeConfig(max_steps=100)
        env = SatelliteEnv(
            episode_config=config,
            sat_config=SatelliteConfig(task_arrival_rate=2.0),
        )
        obs, _ = env.reset(seed=42)
        scheduler = PriorityScheduler()

        anomaly_selections = 0
        total_actions = 0
        for _ in range(100):
            action = scheduler.select_action(obs)
            if action == int(TaskType.ANOMALY):
                anomaly_selections += 1
            if action != len(TaskType):
                total_actions += 1
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        # Priority should select ANOMALY frequently (it's always highest weighted)
        if total_actions > 0:
            anomaly_frac = anomaly_selections / total_actions
            assert anomaly_frac > 0.1  # Should select it at least sometimes
        env.close()
