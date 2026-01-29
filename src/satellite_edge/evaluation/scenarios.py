"""Scenario definitions for evaluation benchmarks.

Defines four key scenarios from the project plan:
1. Steady State (baseline) - moderate resource pressure
2. Event Surge (disaster response) - burst arrivals
3. Degraded Compute (thermal throttling) - reduced capacity
4. Long Blackout (no ground contact) - extended isolation
"""

from dataclasses import dataclass
from typing import Any

from satellite_edge.environment.satellite_env import (
    SatelliteConfig,
    EpisodeConfig,
    SatelliteEnv,
)
from satellite_edge.orbital.ground_station import GroundStationNetwork
from satellite_edge.orbital.orbit_propagator import SimplifiedOrbitModel


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""

    name: str
    description: str
    sat_config: SatelliteConfig
    episode_config: EpisodeConfig
    # Optional: custom orbit model parameters
    orbit_config: dict[str, Any] | None = None

    def create_env(self, seed: int | None = None) -> SatelliteEnv:
        """Create environment instance for this scenario."""
        return create_scenario_env(self, seed=seed)


def _create_steady_state() -> ScenarioConfig:
    """Steady State scenario - moderate resource contention.

    Baseline scenario with ~1.09x contention ratio. Scheduling matters
    but isn't critically constrained.
    """
    return ScenarioConfig(
        name="steady_state",
        description="Moderate resource pressure (~1.09x contention)",
        sat_config=SatelliteConfig(
            compute_capacity=4.0,  # Reduced from 32 to create contention
            buffer_capacity=256.0,
            task_arrival_rate=12.0,  # Higher arrival rate
            priority_event_prob=0.05,
        ),
        episode_config=EpisodeConfig(
            max_steps=500,
            timestep_duration=10.0,
        ),
    )


def _create_event_surge() -> ScenarioConfig:
    """Event Surge scenario - disaster response conditions.

    Simulates emergency imagery processing with burst arrivals and
    elevated priority event probability.
    """
    return ScenarioConfig(
        name="event_surge",
        description="Disaster response: 2x arrival rate, 6x priority events",
        sat_config=SatelliteConfig(
            compute_capacity=4.0,
            buffer_capacity=256.0,
            task_arrival_rate=25.0,  # ~2x normal
            priority_event_prob=0.30,  # 6x normal priority events
        ),
        episode_config=EpisodeConfig(
            max_steps=500,
            timestep_duration=10.0,
        ),
    )


def _create_degraded_compute() -> ScenarioConfig:
    """Degraded Compute scenario - thermal throttling.

    Simulates thermal constraints reducing compute capacity to 50%.
    Tests policy resilience to resource degradation.
    """
    return ScenarioConfig(
        name="degraded_compute",
        description="Thermal throttling: 50% compute capacity",
        sat_config=SatelliteConfig(
            compute_capacity=2.0,  # 50% reduction from baseline
            buffer_capacity=256.0,
            task_arrival_rate=12.0,  # Normal arrivals
            priority_event_prob=0.05,
        ),
        episode_config=EpisodeConfig(
            max_steps=500,
            timestep_duration=10.0,
        ),
    )


def _create_long_blackout() -> ScenarioConfig:
    """Long Blackout scenario - extended no ground contact.

    Simulates extended period without downlink opportunity.
    Buffer fills, requiring aggressive prioritization.
    """
    return ScenarioConfig(
        name="long_blackout",
        description="Extended blackout: reduced contact windows",
        sat_config=SatelliteConfig(
            compute_capacity=4.0,
            buffer_capacity=128.0,  # Smaller buffer to accelerate pressure
            task_arrival_rate=12.0,
            priority_event_prob=0.05,
        ),
        episode_config=EpisodeConfig(
            max_steps=500,
            timestep_duration=10.0,
        ),
        orbit_config={
            "contact_fraction": 0.05,  # Reduced from 15% to 5%
            "contact_duration": 300.0,  # 5 min instead of 10 min
        },
    )


# Pre-defined scenarios for benchmarking
SCENARIOS: dict[str, ScenarioConfig] = {
    "steady_state": _create_steady_state(),
    "event_surge": _create_event_surge(),
    "degraded_compute": _create_degraded_compute(),
    "long_blackout": _create_long_blackout(),
}


def create_scenario_env(
    scenario: ScenarioConfig,
    seed: int | None = None,
) -> SatelliteEnv:
    """Create environment instance for a scenario.

    Args:
        scenario: Scenario configuration.
        seed: Random seed for environment.

    Returns:
        Configured SatelliteEnv instance.
    """
    # Create custom orbit model if specified
    ground_network = None
    if scenario.orbit_config:
        ground_network = GroundStationNetwork()
        # The orbit model will be created internally by SatelliteEnv
        # but we can pass the ground network with custom settings

    env = SatelliteEnv(
        sat_config=scenario.sat_config,
        episode_config=scenario.episode_config,
        ground_network=ground_network,
        seed=seed,
    )

    # Apply custom orbit config if specified
    if scenario.orbit_config and hasattr(env, "_orbit"):
        if isinstance(env._orbit, SimplifiedOrbitModel):
            for key, value in scenario.orbit_config.items():
                if hasattr(env._orbit, key):
                    setattr(env._orbit, key, value)

    return env


def get_scenario(name: str) -> ScenarioConfig:
    """Get scenario by name.

    Args:
        name: Scenario name.

    Returns:
        ScenarioConfig instance.

    Raises:
        KeyError: If scenario not found.
    """
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())
