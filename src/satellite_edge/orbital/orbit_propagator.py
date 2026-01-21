"""Orbit propagation and ground station visibility using Skyfield."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time, Timescale

from satellite_edge.orbital.ground_station import GroundStation, GroundStationNetwork


# ISS-like LEO TLE for realistic orbital dynamics
# This gives ~92 minute period, ~400km altitude
DEFAULT_TLE = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9025",
    "2 25544  51.6400 208.9163 0006703  30.0000 330.0000 15.49000000000000"
)


@dataclass
class VisibilityWindow:
    """A window of ground station visibility."""
    station: GroundStation
    start_time: float       # seconds from epoch
    end_time: float         # seconds from epoch
    max_elevation: float    # peak elevation in degrees

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class OrbitModel(Protocol):
    """Protocol for orbit propagation models."""

    def get_ground_contact(self, time: float) -> tuple[bool, GroundStation | None, float]:
        """Check if satellite is in contact with any ground station.

        Args:
            time: Simulation time in seconds from epoch

        Returns:
            Tuple of (in_contact, station, elevation_degrees)
        """
        ...

    def get_time_to_next_contact(self, time: float) -> float:
        """Get seconds until next ground station contact."""
        ...

    def get_subsatellite_point(self, time: float) -> tuple[float, float]:
        """Get current subsatellite point (lat, lon in degrees)."""
        ...


class OrbitPropagator:
    """Skyfield-based orbit propagator for LEO satellites.

    Computes satellite position and ground station visibility using
    SGP4 propagation from TLE elements.
    """

    def __init__(
        self,
        tle_line1: str = DEFAULT_TLE[0],
        tle_line2: str = DEFAULT_TLE[1],
        ground_network: GroundStationNetwork | None = None,
        ts: Timescale | None = None,
    ):
        self._ts = ts or load.timescale()
        self._satellite = EarthSatellite(tle_line1, tle_line2, "SATELLITE", self._ts)
        self._network = ground_network or GroundStationNetwork()

        # Cache epoch for time conversions
        self._epoch = self._satellite.epoch

        # Pre-compute visibility windows for efficiency
        self._visibility_cache: list[VisibilityWindow] = []
        self._cache_end_time: float = 0.0

    def _time_to_skyfield(self, sim_time: float) -> Time:
        """Convert simulation time (seconds from epoch) to Skyfield Time."""
        # Add seconds to epoch Julian date
        jd = self._epoch.tt + sim_time / 86400.0
        return self._ts.tt_jd(jd)

    def get_position(self, time: float) -> tuple[float, float, float]:
        """Get satellite ECEF position at given time.

        Returns:
            (x, y, z) in kilometers
        """
        t = self._time_to_skyfield(time)
        geocentric = self._satellite.at(t)
        return tuple(geocentric.position.km)

    def get_subsatellite_point(self, time: float) -> tuple[float, float]:
        """Get subsatellite point (latitude, longitude) in degrees."""
        t = self._time_to_skyfield(time)
        geocentric = self._satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        return subpoint.latitude.degrees, subpoint.longitude.degrees

    def get_ground_contact(
        self, time: float
    ) -> tuple[bool, GroundStation | None, float]:
        """Check if satellite is visible from any ground station.

        Returns:
            (in_contact, station, elevation_degrees)
            If multiple stations visible, returns the one with highest elevation.
        """
        t = self._time_to_skyfield(time)
        best_station = None
        best_elevation = 0.0

        for station in self._network:
            difference = self._satellite - station.to_skyfield()
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            elevation = alt.degrees

            if elevation >= station.min_elevation and elevation > best_elevation:
                best_station = station
                best_elevation = elevation

        return (best_station is not None, best_station, best_elevation)

    def _compute_visibility_windows(
        self, start_time: float, duration: float
    ) -> list[VisibilityWindow]:
        """Compute all visibility windows in a time range.

        Uses Skyfield's find_events for accurate rise/set times.
        """
        windows = []
        t0 = self._time_to_skyfield(start_time)
        t1 = self._time_to_skyfield(start_time + duration)

        for station in self._network:
            topos = station.to_skyfield()
            times, events = self._satellite.find_events(
                topos, t0, t1, altitude_degrees=station.min_elevation
            )

            # Process events: 0=rise, 1=culminate, 2=set
            rise_time = None
            max_elev = 0.0

            for ti, event in zip(times, events):
                event_sim_time = (ti.tt - self._epoch.tt) * 86400.0

                if event == 0:  # rise
                    rise_time = event_sim_time
                    max_elev = 0.0
                elif event == 1:  # culminate
                    diff = self._satellite - topos
                    alt, _, _ = diff.at(ti).altaz()
                    max_elev = alt.degrees
                elif event == 2 and rise_time is not None:  # set
                    windows.append(VisibilityWindow(
                        station=station,
                        start_time=rise_time,
                        end_time=event_sim_time,
                        max_elevation=max_elev
                    ))
                    rise_time = None

        # Sort by start time
        windows.sort(key=lambda w: w.start_time)
        return windows

    def get_time_to_next_contact(self, time: float, lookahead: float = 7200.0) -> float:
        """Get seconds until next ground station contact.

        Args:
            time: Current simulation time
            lookahead: How far ahead to search (seconds), default 2 hours

        Returns:
            Seconds until contact, or lookahead if none found
        """
        # Check if currently in contact
        in_contact, _, _ = self.get_ground_contact(time)
        if in_contact:
            return 0.0

        # Extend cache if needed
        if time + lookahead > self._cache_end_time:
            new_windows = self._compute_visibility_windows(
                self._cache_end_time, lookahead + 3600.0
            )
            self._visibility_cache.extend(new_windows)
            self._cache_end_time = time + lookahead + 3600.0

        # Find next window
        for window in self._visibility_cache:
            if window.start_time > time:
                return window.start_time - time
            if window.start_time <= time <= window.end_time:
                return 0.0

        return lookahead

    def get_orbital_period(self) -> float:
        """Get orbital period in seconds."""
        # Mean motion is in revolutions per day
        mean_motion = self._satellite.model.no_kozai  # rad/min
        if mean_motion > 0:
            return 2 * np.pi / mean_motion * 60  # seconds
        return 5400.0  # default ~90 min

    def is_sunlit(self, time: float) -> bool:
        """Check if satellite is in sunlight (not eclipsed)."""
        t = self._time_to_skyfield(time)
        eph = load('de421.bsp')
        return self._satellite.at(t).is_sunlit(eph)


class SimplifiedOrbitModel:
    """Simplified orbit model for faster simulation.

    Uses sinusoidal approximation instead of full SGP4 propagation.
    Useful for rapid prototyping and testing.
    """

    def __init__(
        self,
        orbital_period: float = 5400.0,  # 90 minutes
        contact_fraction: float = 0.15,   # ~15% of orbit in contact
        contact_duration: float = 600.0,  # 10 minute typical pass
        ground_network: GroundStationNetwork | None = None,
    ):
        self.orbital_period = orbital_period
        self.contact_fraction = contact_fraction
        self.contact_duration = contact_duration
        self._network = ground_network or GroundStationNetwork()

        # Compute contact timing
        self._contacts_per_orbit = max(1, int(
            orbital_period * contact_fraction / contact_duration
        ))

    def get_ground_contact(
        self, time: float
    ) -> tuple[bool, GroundStation | None, float]:
        """Simplified contact check based on orbital phase."""
        phase = (time % self.orbital_period) / self.orbital_period

        # Create contact windows at regular intervals
        for i in range(self._contacts_per_orbit):
            window_center = (i + 0.5) / self._contacts_per_orbit
            window_half = (self.contact_duration / 2) / self.orbital_period

            if abs(phase - window_center) < window_half:
                # In contact - pick station based on orbit number
                station_idx = int(time / self.orbital_period) % len(self._network.stations)
                station = self._network.stations[station_idx]
                # Approximate elevation as parabolic within window
                rel_pos = (phase - window_center) / window_half
                elevation = station.min_elevation + (90 - station.min_elevation) * (1 - rel_pos**2)
                return True, station, elevation

        return False, None, 0.0

    def get_time_to_next_contact(self, time: float) -> float:
        """Get time to next contact window."""
        phase = (time % self.orbital_period) / self.orbital_period

        for i in range(self._contacts_per_orbit):
            window_center = (i + 0.5) / self._contacts_per_orbit
            window_start = window_center - (self.contact_duration / 2) / self.orbital_period

            if window_start > phase:
                return (window_start - phase) * self.orbital_period

        # Next contact is in next orbit
        first_window_start = 0.5 / self._contacts_per_orbit - \
            (self.contact_duration / 2) / self.orbital_period
        return (1.0 - phase + first_window_start) * self.orbital_period

    def get_subsatellite_point(self, time: float) -> tuple[float, float]:
        """Approximate subsatellite point using sinusoidal model."""
        phase = (time % self.orbital_period) / self.orbital_period

        # ISS-like inclination ~51.6 degrees
        lat = 51.6 * np.sin(2 * np.pi * phase)

        # Longitude progresses ~22.5 deg per orbit (nodal regression)
        orbits = time / self.orbital_period
        lon = (phase * 360 - orbits * 22.5) % 360 - 180

        return lat, lon
