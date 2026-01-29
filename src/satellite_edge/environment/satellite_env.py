"""Gymnasium-compatible satellite edge computing environment."""

from dataclasses import dataclass
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from satellite_edge.environment.tasks import (
    Task,
    ImageTask,
    TaskType,
    TaskQueue,
    TASK_SPECS,
)
from satellite_edge.orbital.orbit_propagator import (
    SimplifiedOrbitModel,
    OrbitPropagator,
)
from satellite_edge.orbital.ground_station import GroundStationNetwork


@dataclass
class SatelliteConfig:
    """Configuration for satellite compute and storage resources.

    Grounded in real hardware specs:
    - Jiguang 1000: ~32 TOPS AI compute
    - Typical LEO sat: 256-512 GB onboard storage
    - Power budget: 100-500W depending on solar/battery
    """

    compute_capacity: float = 32.0  # TOPS available per timestep
    buffer_capacity: float = 256.0  # GB onboard storage
    power_capacity: float = 300.0  # Watts available
    power_per_tops: float = 5.0  # W per TOPS used

    # Task generation parameters
    task_arrival_rate: float = 0.5  # Mean tasks per timestep
    priority_event_prob: float = 0.05  # Probability of priority boost event

    # Image-aware scheduling
    use_image_features: bool = False  # Extend obs space with image metrics


@dataclass
class EpisodeConfig:
    """Configuration for episode structure."""

    max_steps: int = 1000  # Steps per episode
    timestep_duration: float = 10.0  # Seconds per simulation step
    use_full_orbit_model: bool = False  # Use Skyfield vs simplified


class SatelliteEnv(gym.Env):
    """Gymnasium environment for satellite edge compute scheduling.

    The agent must allocate limited onboard compute across competing CV tasks
    while managing buffer constraints and ground station downlink windows.

    Observation Space (14 dims, or 22 if use_image_features=True):
        Base features (14 dims):
        - compute_available: Fraction of compute budget remaining [0, 1]
        - buffer_usage: Fraction of buffer capacity used [0, 1]
        - queue_depths: Tasks per type, normalized [0, 1] x num_task_types
        - queue_compute: Total compute per type, normalized [0, 1] x num_task_types
        - ground_contact: Binary indicator of ground station visibility
        - time_to_contact: Normalized time until next contact [0, 1]
        - contact_duration: Normalized remaining contact time [0, 1]
        - power_available: Fraction of power budget remaining [0, 1]

        Image features (8 dims, when use_image_features=True):
        - avg_cloud: Mean cloud_fraction per task type [0, 1] x num_task_types
        - avg_quality: Mean image_quality_multiplier per task type [0, 1] x num_task_types

    Action Space:
        Discrete(num_task_types + 1):
        - 0-3: Prioritize task type (DETECTION, ANOMALY, CLOUD_MASK, COMPRESSION)
        - 4: IDLE (save power/compute for later)

    Reward:
        - Positive: Value from completed tasks (time-decayed)
        - Negative: Buffer overflow penalty
        - Negative: Missed downlink opportunity penalty
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        sat_config: SatelliteConfig | None = None,
        episode_config: EpisodeConfig | None = None,
        ground_network: GroundStationNetwork | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()

        self.sat_config = sat_config or SatelliteConfig()
        self.episode_config = episode_config or EpisodeConfig()
        self.render_mode = render_mode

        # Initialize orbital model
        self._ground_network = ground_network or GroundStationNetwork()
        if self.episode_config.use_full_orbit_model:
            self._orbit = OrbitPropagator(ground_network=self._ground_network)
        else:
            self._orbit = SimplifiedOrbitModel(ground_network=self._ground_network)

        # Define spaces
        self._num_task_types = len(TaskType)

        # Observation: flat Box for SB3 compatibility
        # Base: [compute_avail, buffer_usage, queue_depths x4, queue_compute x4,
        #        ground_contact, time_to_contact, contact_duration, power_avail]
        # = 2 + 2*4 + 4 = 14
        # Extended (use_image_features=True): + avg_cloud x4 + avg_quality x4 = 22
        base_obs_dim = 2 + 2 * self._num_task_types + 4
        if self.sat_config.use_image_features:
            obs_dim = base_obs_dim + 2 * self._num_task_types  # +8 for image features
        else:
            obs_dim = base_obs_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: prioritize a task type or idle
        self.action_space = spaces.Discrete(self._num_task_types + 1)

        # State variables (initialized in reset)
        self._task_queue: TaskQueue = TaskQueue()
        self._buffer_used: float = 0.0
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._compute_used_this_step: float = 0.0
        self._power_used_this_step: float = 0.0

        # Metrics for info dict
        self._total_value: float = 0.0
        self._tasks_completed: int = 0
        self._tasks_dropped: int = 0
        self._downlink_data: float = 0.0

        # Extended metrics for Phase 4 evaluation
        self._task_completion_records: list[dict] = []
        self._completions_by_type: dict[int, int] = {}
        self._value_by_type: dict[int, float] = {}
        self._buffer_overflow_count: int = 0
        self._contact_time_used: float = 0.0
        self._contact_time_available: float = 0.0
        self._potential_value: float = 0.0  # Value if all tasks completed instantly

        # Contact state tracking
        self._in_contact: bool = False
        self._contact_start_time: float = 0.0
        self._current_station = None

    def _get_obs(self) -> np.ndarray:
        """Build observation vector.

        Base observation (14 dims):
            obs[0]     compute_avail         fraction of compute capacity remaining
            obs[1]     buffer_usage          fraction of buffer capacity used
            obs[2:6]   queue_depths[4]       task count per type / max_queue_size
            obs[6:10]  queue_compute[4]      total compute per type / (capacity * 10)
            obs[10]    ground_contact        1.0 if in contact, 0.0 else
            obs[11]    time_to_contact       normalized by orbital period
            obs[12]    contact_duration      fraction of remaining contact time
            obs[13]    power_avail           fraction of power capacity remaining

        Extended observation (22 dims, when use_image_features=True):
            obs[14:18] avg_cloud[4]          mean cloud_fraction per task type [0,1]
            obs[18:22] avg_quality[4]        mean image_quality_multiplier per task type [0,1]
        """
        # Compute availability (replenishes each step)
        compute_avail = (
            1.0 - self._compute_used_this_step / self.sat_config.compute_capacity
        )

        # Buffer usage
        buffer_usage = self._buffer_used / self.sat_config.buffer_capacity

        # Queue statistics
        queue_depths = self._task_queue.get_queue_depths() / self._task_queue.max_size
        queue_compute = self._task_queue.get_total_compute_by_type()
        max_compute = self.sat_config.compute_capacity * 10  # Normalize
        queue_compute = np.clip(queue_compute / max_compute, 0, 1)

        # Ground contact status
        in_contact, station, elevation = self._orbit.get_ground_contact(self._sim_time)
        ground_contact = 1.0 if in_contact else 0.0

        # Time to next contact (normalized by orbital period)
        if in_contact:
            time_to_contact = 0.0
        else:
            time_to_next = self._orbit.get_time_to_next_contact(self._sim_time)
            orbital_period = getattr(self._orbit, "orbital_period", 5400.0)
            time_to_contact = np.clip(time_to_next / orbital_period, 0, 1)

        # Contact duration remaining (if in contact)
        if in_contact and self._contact_start_time > 0:
            contact_elapsed = self._sim_time - self._contact_start_time
            typical_duration = 600.0  # 10 min typical pass
            contact_duration = np.clip(1.0 - contact_elapsed / typical_duration, 0, 1)
        else:
            contact_duration = 0.0

        # Power availability
        power_avail = 1.0 - self._power_used_this_step / self.sat_config.power_capacity

        # Build base observation
        base_obs = [
            compute_avail,
            buffer_usage,
            *queue_depths,
            *queue_compute,
            ground_contact,
            time_to_contact,
            contact_duration,
            power_avail,
        ]

        # Add image features if enabled
        if self.sat_config.use_image_features:
            avg_cloud = self._task_queue.get_avg_cloud_by_type()
            avg_quality = self._task_queue.get_avg_quality_by_type()
            obs = np.array(
                [*base_obs, *avg_cloud, *avg_quality],
                dtype=np.float32,
            )
        else:
            obs = np.array(base_obs, dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with metrics."""
        contact_efficiency = 0.0
        if self._contact_time_available > 1e-6:
            contact_efficiency = self._contact_time_used / self._contact_time_available

        return {
            "sim_time": self._sim_time,
            "step": self._step_count,
            "total_value": self._total_value,
            "tasks_completed": self._tasks_completed,
            "tasks_dropped": self._tasks_dropped,
            "queue_size": len(self._task_queue),
            "buffer_used": self._buffer_used,
            "downlink_data": self._downlink_data,
            "in_contact": self._in_contact,
            # Extended metrics for Phase 4 evaluation
            "task_completions": self._task_completion_records,
            "completions_by_type": self._completions_by_type.copy(),
            "value_by_type": self._value_by_type.copy(),
            "buffer_overflows": self._buffer_overflow_count,
            "contact_time_used": self._contact_time_used,
            "contact_time_available": self._contact_time_available,
            "contact_efficiency": contact_efficiency,
            "potential_value": self._potential_value,
        }

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset state
        self._task_queue = TaskQueue()
        self._buffer_used = 0.0
        self._sim_time = 0.0
        self._step_count = 0
        self._compute_used_this_step = 0.0
        self._power_used_this_step = 0.0

        # Reset metrics
        self._total_value = 0.0
        self._tasks_completed = 0
        self._tasks_dropped = 0
        self._downlink_data = 0.0

        # Reset extended metrics
        self._task_completion_records = []
        self._completions_by_type = {}
        self._value_by_type = {}
        self._buffer_overflow_count = 0
        self._contact_time_used = 0.0
        self._contact_time_available = 0.0
        self._potential_value = 0.0

        # Reset contact tracking
        self._in_contact = False
        self._contact_start_time = 0.0
        self._current_station = None

        # Generate initial tasks
        initial_tasks = self.np_random.integers(3, 8)
        for _ in range(initial_tasks):
            self._generate_task()

        return self._get_obs(), self._get_info()

    def _generate_task(self) -> Task | None:
        """Generate a new random task and add to queue.

        When use_image_features is enabled, creates ImageTask with synthetic
        image metrics. Otherwise creates basic Task.
        """
        # Sample task type (weighted towards detection and compression)
        weights = [0.35, 0.15, 0.20, 0.30]  # DET, ANO, CLOUD, COMP
        task_type = TaskType(self.np_random.choice(len(TaskType), p=weights))

        # Priority boost for special events
        priority = 1.0
        if self.np_random.random() < self.sat_config.priority_event_prob:
            priority = self.np_random.uniform(2.0, 5.0)

        tile_id = self.np_random.integers(0, 10000)

        if self.sat_config.use_image_features:
            # Generate synthetic image metrics
            # Cloud fraction: Beta(2, 5) skews toward clear sky (~0.28 mean)
            cloud_fraction = float(self.np_random.beta(2, 5))
            # Entropy: Uniform spread representing varied scene complexity
            entropy_score = float(self.np_random.uniform(0.3, 0.9))
            # Edge density: Uniform spread for feature richness
            edge_density = float(self.np_random.uniform(0.2, 0.8))

            task = ImageTask(
                task_type=task_type,
                tile_id=tile_id,
                arrival_time=self._sim_time,
                priority_boost=priority,
                cloud_fraction=cloud_fraction,
                entropy_score=entropy_score,
                edge_density=edge_density,
            )
        else:
            task = Task(
                task_type=task_type,
                tile_id=tile_id,
                arrival_time=self._sim_time,
                priority_boost=priority,
            )

        # Check buffer capacity
        if (
            self._buffer_used + task.spec.memory_footprint
            > self.sat_config.buffer_capacity
        ):
            self._tasks_dropped += 1
            return None

        if self._task_queue.add(task):
            self._buffer_used += task.spec.memory_footprint
            # Track potential value (value if completed instantly)
            self._potential_value += task.current_value(self._sim_time)
            return task

        self._tasks_dropped += 1
        return None

    def _process_tasks(self, action: int) -> float:
        """Process tasks based on agent action, return reward."""
        reward = 0.0
        compute_remaining = self.sat_config.compute_capacity
        power_remaining = self.sat_config.power_capacity

        if action == self._num_task_types:
            # IDLE action - no compute used
            self._compute_used_this_step = 0.0
            self._power_used_this_step = 0.0
            return 0.0

        # Get prioritized task type
        prioritized_type = TaskType(action)

        # Process tasks in priority order
        # 1. All tasks of prioritized type
        # 2. Then other types in round-robin

        processing_order = [prioritized_type] + [
            t for t in TaskType if t != prioritized_type
        ]

        for task_type in processing_order:
            tasks = self._task_queue.get_by_type(task_type)

            for task in tasks:
                if compute_remaining <= 0 or power_remaining <= 0:
                    break

                # Check power constraint
                power_needed = task.remaining_compute * self.sat_config.power_per_tops
                if power_needed > power_remaining:
                    continue

                # Apply compute to task
                tops_used = task.apply_compute(
                    compute_remaining, self.episode_config.timestep_duration
                )
                compute_remaining -= tops_used
                power_remaining -= tops_used * self.sat_config.power_per_tops

        # Track resource usage
        self._compute_used_this_step = (
            self.sat_config.compute_capacity - compute_remaining
        )
        self._power_used_this_step = self.sat_config.power_capacity - power_remaining

        # Collect completed tasks and calculate reward
        completed = self._task_queue.remove_completed()
        for task in completed:
            value = task.current_value(self._sim_time)
            reward += value
            self._total_value += value
            self._tasks_completed += 1
            self._buffer_used -= task.spec.memory_footprint

            # Track completion record for latency analysis
            task_type_int = int(task.task_type)
            self._task_completion_records.append(
                {
                    "task_type": task_type_int,
                    "arrival_time": task.arrival_time,
                    "completion_time": self._sim_time,
                    "priority": task.priority_boost,
                    "value": value,
                }
            )

            # Track per-type completions and value
            self._completions_by_type[task_type_int] = (
                self._completions_by_type.get(task_type_int, 0) + 1
            )
            self._value_by_type[task_type_int] = (
                self._value_by_type.get(task_type_int, 0.0) + value
            )

        # Prevent floating point precision issues
        self._buffer_used = max(0.0, self._buffer_used)

        return reward

    def _update_contact_state(self) -> float:
        """Update ground contact state and handle downlink, return reward."""
        reward = 0.0
        in_contact, station, elevation = self._orbit.get_ground_contact(self._sim_time)

        # Contact state transition
        if in_contact and not self._in_contact:
            # Contact acquired
            self._contact_start_time = self._sim_time
            self._current_station = station
        elif not in_contact and self._in_contact:
            # Contact lost
            self._current_station = None

        self._in_contact = in_contact

        # Downlink during contact
        if in_contact and station is not None:
            # Downlink processed data (compression tasks)
            downlink_rate = station.downlink_rate  # Gbps
            data_downlinked = (
                downlink_rate * self.episode_config.timestep_duration / 8
            )  # GB
            actual_downlink = min(
                data_downlinked, self._buffer_used * 0.5
            )  # Cap at half buffer
            self._buffer_used = max(0, self._buffer_used - actual_downlink)
            self._downlink_data += actual_downlink

            # Track contact time for efficiency calculation
            self._contact_time_available += self.episode_config.timestep_duration
            if actual_downlink > 0:
                # Proportion of timestep actually used for downlink
                self._contact_time_used += self.episode_config.timestep_duration * (
                    actual_downlink / data_downlinked
                )

            # Small reward for successful downlink
            reward += actual_downlink * 0.1

        return reward

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep.

        Args:
            action: Task type to prioritize (0-3) or IDLE (4)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Advance simulation time
        self._sim_time += self.episode_config.timestep_duration
        self._step_count += 1

        # Reset per-step resource tracking
        self._compute_used_this_step = 0.0
        self._power_used_this_step = 0.0

        # Generate new tasks (Poisson arrival)
        num_arrivals = self.np_random.poisson(self.sat_config.task_arrival_rate)
        for _ in range(num_arrivals):
            self._generate_task()

        # Process tasks based on action
        task_reward = self._process_tasks(action)

        # Update ground contact and handle downlink
        contact_reward = self._update_contact_state()

        # Buffer overflow penalty
        overflow_penalty = 0.0
        if self._buffer_used > self.sat_config.buffer_capacity:
            overflow = self._buffer_used - self.sat_config.buffer_capacity
            overflow_penalty = -overflow * 2.0
            self._buffer_used = self.sat_config.buffer_capacity
            self._buffer_overflow_count += 1

        # Total reward
        reward = task_reward + contact_reward + overflow_penalty

        # Check termination
        terminated = False
        truncated = self._step_count >= self.episode_config.max_steps

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render current state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            lines = [
                f"=== Step {self._step_count} | Time: {self._sim_time:.0f}s ===",
                f"Queue: {len(self._task_queue)} tasks | Buffer: {self._buffer_used:.1f}/{self.sat_config.buffer_capacity:.0f} GB",
                f"Contact: {'YES' if self._in_contact else 'NO'} | Value: {self._total_value:.1f}",
                f"Completed: {self._tasks_completed} | Dropped: {self._tasks_dropped}",
            ]
            output = "\n".join(lines)

            if self.render_mode == "human":
                print(output)
                return None
            return output

        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
