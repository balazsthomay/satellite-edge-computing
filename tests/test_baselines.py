"""Tests for baseline scheduling policies."""

import numpy as np
import pytest

from satellite_edge.agents.baselines import (
    BaseScheduler,
    FIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    GreedyComputeScheduler,
    RandomScheduler,
)
from satellite_edge.environment.satellite_env import SatelliteEnv, SatelliteConfig, EpisodeConfig
from satellite_edge.environment.tasks import TaskType, TASK_SPECS


# ─── Observation fixtures ───────────────────────────────────────────────

def make_obs(
    queue_depths: list[float] | None = None,
    compute_avail: float = 1.0,
    buffer_usage: float = 0.2,
    power_avail: float = 1.0,
) -> np.ndarray:
    """Create a mock observation vector for testing schedulers."""
    if queue_depths is None:
        queue_depths = [0.1, 0.05, 0.1, 0.1]  # normalized by 50

    # obs[0:2] = compute_avail, buffer_usage
    # obs[2:6] = queue_depths
    # obs[6:10] = queue_compute (not used by baselines, set to match depths)
    # obs[10:14] = ground_contact, time_to_contact, contact_duration, power_avail
    queue_compute = [d * 0.5 for d in queue_depths]  # arbitrary
    obs = np.array(
        [compute_avail, buffer_usage]
        + queue_depths
        + queue_compute
        + [0.0, 0.5, 0.0, power_avail],
        dtype=np.float32,
    )
    return obs


# ─── FIFOScheduler tests ────────────────────────────────────────────────

class TestFIFOScheduler:
    def test_selects_deepest_queue(self):
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.1, 0.3, 0.05, 0.2])
        assert sched.select_action(obs) == 1  # ANOMALY has deepest queue

    def test_idles_on_empty_queues(self):
        sched = FIFOScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == len(TaskType)

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
    def test_selects_highest_value_type(self):
        sched = PriorityScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])
        # ANOMALY: 15 * 0.9 = 13.5 > DETECTION: 10 * 0.8 = 8.0
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_skips_empty_high_priority(self):
        sched = PriorityScheduler()
        obs = make_obs(queue_depths=[0.1, 0.0, 0.1, 0.1])
        # ANOMALY empty, should pick next: DETECTION (8.0)
        assert sched.select_action(obs) == int(TaskType.DETECTION)

    def test_idles_on_empty(self):
        sched = PriorityScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == len(TaskType)

    def test_priority_ordering(self):
        """Verify the static priority ordering is correct."""
        sched = PriorityScheduler()
        # ANOMALY (13.5) > DETECTION (8.0) > COMPRESSION (2.5) > CLOUD_MASK (0.9)
        expected_order = [TaskType.ANOMALY, TaskType.DETECTION, TaskType.COMPRESSION, TaskType.CLOUD_MASK]
        assert sched._priority_order == expected_order

    def test_only_cloud_mask_available(self):
        sched = PriorityScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.2, 0.0])
        assert sched.select_action(obs) == int(TaskType.CLOUD_MASK)


# ─── RoundRobinScheduler tests ─────────────────────────────────────────

class TestRoundRobinScheduler:
    def test_cycles_through_types(self):
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])

        actions = [sched.select_action(obs) for _ in range(8)]
        # Should cycle: 0, 1, 2, 3, 0, 1, 2, 3
        assert actions == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_skips_empty_types(self):
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.1, 0.0, 0.1, 0.0])

        actions = [sched.select_action(obs) for _ in range(4)]
        # Skips 1, 3: should cycle 0, 2, 0, 2
        assert actions == [0, 2, 0, 2]

    def test_reset_restarts_cycle(self):
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])

        sched.select_action(obs)  # 0
        sched.select_action(obs)  # 1
        sched.reset()
        assert sched.select_action(obs) == 0  # Back to start

    def test_all_empty_idles(self):
        sched = RoundRobinScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == len(TaskType)


# ─── GreedyComputeScheduler tests ──────────────────────────────────────

class TestGreedyComputeScheduler:
    def test_selects_cheapest_type(self):
        sched = GreedyComputeScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.1, 0.1])
        # CLOUD_MASK (1.5) < ANOMALY (3.0) < COMPRESSION (4.0) < DETECTION (6.0)
        assert sched.select_action(obs) == int(TaskType.CLOUD_MASK)

    def test_skips_empty_cheapest(self):
        sched = GreedyComputeScheduler()
        obs = make_obs(queue_depths=[0.1, 0.1, 0.0, 0.1])
        # CLOUD_MASK empty, pick ANOMALY (next cheapest)
        assert sched.select_action(obs) == int(TaskType.ANOMALY)

    def test_idles_on_empty(self):
        sched = GreedyComputeScheduler()
        obs = make_obs(queue_depths=[0.0, 0.0, 0.0, 0.0])
        assert sched.select_action(obs) == len(TaskType)

    def test_cost_ordering(self):
        sched = GreedyComputeScheduler()
        expected = [TaskType.CLOUD_MASK, TaskType.ANOMALY, TaskType.COMPRESSION, TaskType.DETECTION]
        assert sched._cost_order == expected


# ─── RandomScheduler tests ──────────────────────────────────────────────

class TestRandomScheduler:
    def test_actions_in_valid_range(self):
        sched = RandomScheduler(seed=42)
        obs = make_obs()
        actions = [sched.select_action(obs) for _ in range(100)]
        assert all(0 <= a <= len(TaskType) for a in actions)

    def test_produces_all_actions(self):
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

        for scheduler_cls in [FIFOScheduler, PriorityScheduler, RoundRobinScheduler, GreedyComputeScheduler]:
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
            sat_config=SatelliteConfig(task_arrival_rate=2.0),  # More tasks = more contention
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

        # Priority should select ANOMALY whenever it's available (highest value*urgency)
        if total_actions > 0:
            anomaly_frac = anomaly_selections / total_actions
            assert anomaly_frac > 0.05  # Should select it at least sometimes
        env.close()
