"""Ground station modeling for satellite communications."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyfield.api import wgs84

if TYPE_CHECKING:
    from skyfield.toposlib import GeographicPosition


@dataclass(frozen=True)
class GroundStation:
    """A ground station for satellite downlink.

    Attributes:
        name: Human-readable identifier
        latitude: Degrees north (negative for south)
        longitude: Degrees east (negative for west)
        min_elevation: Minimum elevation angle for contact (degrees)
        downlink_rate: Maximum data rate during contact (Gbps)
    """
    name: str
    latitude: float
    longitude: float
    min_elevation: float = 10.0  # degrees above horizon
    downlink_rate: float = 1.0   # Gbps

    def to_skyfield(self) -> "GeographicPosition":
        """Convert to Skyfield geographic position."""
        return wgs84.latlon(self.latitude, self.longitude)


# Real ground station locations for Earth observation satellites
# Based on common commercial EO ground station networks
DEFAULT_GROUND_STATIONS = [
    GroundStation("Svalbard", 78.2306, 15.3894, 5.0, 1.5),      # Polar coverage
    GroundStation("Fairbanks", 64.8594, -147.8442, 5.0, 1.2),   # Alaska
    GroundStation("Wallops", 37.9367, -75.4707, 10.0, 1.0),     # Virginia
    GroundStation("Hartebeesthoek", -25.8872, 27.7077, 10.0, 0.8),  # South Africa
    GroundStation("Alice Springs", -23.7009, 133.8731, 10.0, 1.0),  # Australia
    GroundStation("McMurdo", -77.8419, 166.6863, 5.0, 0.6),     # Antarctica
]


class GroundStationNetwork:
    """Network of ground stations for satellite communications.

    Provides methods to check visibility and compute downlink opportunities.
    """

    def __init__(self, stations: list[GroundStation] | None = None):
        self.stations = stations or list(DEFAULT_GROUND_STATIONS)

    def __len__(self) -> int:
        return len(self.stations)

    def __iter__(self):
        return iter(self.stations)

    def get_skyfield_positions(self) -> list["GeographicPosition"]:
        """Get all stations as Skyfield positions."""
        return [s.to_skyfield() for s in self.stations]
