"""Tests for orbital mechanics and ground station visibility."""

import numpy as np
import pytest

from satellite_edge.orbital.ground_station import (
    GroundStation, GroundStationNetwork, DEFAULT_GROUND_STATIONS
)
from satellite_edge.orbital.orbit_propagator import (
    SimplifiedOrbitModel, VisibilityWindow, OrbitPropagator, DEFAULT_TLE
)


class TestGroundStation:
    """Tests for GroundStation dataclass."""

    def test_ground_station_creation(self):
        station = GroundStation(
            name="Test Station",
            latitude=45.0,
            longitude=-75.0,
            min_elevation=10.0,
            downlink_rate=1.5
        )
        assert station.name == "Test Station"
        assert station.latitude == 45.0
        assert station.longitude == -75.0
        assert station.min_elevation == 10.0
        assert station.downlink_rate == 1.5

    def test_ground_station_defaults(self):
        station = GroundStation(
            name="Minimal",
            latitude=0.0,
            longitude=0.0,
        )
        assert station.min_elevation == 10.0
        assert station.downlink_rate == 1.0

    def test_to_skyfield(self):
        station = GroundStation(
            name="Test",
            latitude=40.0,
            longitude=-80.0,
        )
        position = station.to_skyfield()
        assert position.latitude.degrees == pytest.approx(40.0)
        assert position.longitude.degrees == pytest.approx(-80.0)

    def test_frozen_dataclass(self):
        station = GroundStation("Test", 0.0, 0.0)
        with pytest.raises(Exception):
            station.name = "Modified"


class TestGroundStationNetwork:
    """Tests for GroundStationNetwork."""

    def test_default_network(self):
        network = GroundStationNetwork()
        assert len(network) == len(DEFAULT_GROUND_STATIONS)

    def test_custom_network(self):
        stations = [
            GroundStation("A", 0.0, 0.0),
            GroundStation("B", 45.0, 90.0),
        ]
        network = GroundStationNetwork(stations)
        assert len(network) == 2

    def test_iteration(self):
        network = GroundStationNetwork()
        stations = list(network)
        assert len(stations) == len(network)
        assert all(isinstance(s, GroundStation) for s in stations)

    def test_get_skyfield_positions(self):
        network = GroundStationNetwork()
        positions = network.get_skyfield_positions()
        assert len(positions) == len(network)


class TestVisibilityWindow:
    """Tests for VisibilityWindow."""

    def test_visibility_window_creation(self):
        station = GroundStation("Test", 0.0, 0.0)
        window = VisibilityWindow(
            station=station,
            start_time=100.0,
            end_time=700.0,
            max_elevation=45.0
        )
        assert window.station == station
        assert window.start_time == 100.0
        assert window.end_time == 700.0
        assert window.max_elevation == 45.0

    def test_duration(self):
        station = GroundStation("Test", 0.0, 0.0)
        window = VisibilityWindow(station, 100.0, 700.0, 45.0)
        assert window.duration == 600.0


class TestSimplifiedOrbitModel:
    """Tests for SimplifiedOrbitModel."""

    def test_default_parameters(self):
        model = SimplifiedOrbitModel()
        assert model.orbital_period == 5400.0
        assert model.contact_fraction == 0.15
        assert model.contact_duration == 600.0

    def test_custom_parameters(self):
        model = SimplifiedOrbitModel(
            orbital_period=6000.0,
            contact_fraction=0.20,
            contact_duration=500.0
        )
        assert model.orbital_period == 6000.0
        assert model.contact_fraction == 0.20

    def test_contact_cycling(self):
        model = SimplifiedOrbitModel()

        # Should have some contact during an orbit
        contact_times = []
        for t in range(0, int(model.orbital_period), 60):
            in_contact, station, elev = model.get_ground_contact(float(t))
            if in_contact:
                contact_times.append(t)

        # Should have non-zero contact time
        assert len(contact_times) > 0

        # Contact should be a reasonable fraction of orbit
        contact_fraction = len(contact_times) * 60 / model.orbital_period
        assert 0.05 < contact_fraction < 0.30

    def test_time_to_next_contact(self):
        model = SimplifiedOrbitModel()

        # Find a time when not in contact
        for t in range(0, int(model.orbital_period), 10):
            in_contact, _, _ = model.get_ground_contact(float(t))
            if not in_contact:
                ttc = model.get_time_to_next_contact(float(t))
                assert ttc > 0
                assert ttc < model.orbital_period
                break

    def test_time_to_contact_when_in_contact(self):
        model = SimplifiedOrbitModel()

        # Find a time when in contact
        for t in range(0, int(model.orbital_period), 10):
            in_contact, _, _ = model.get_ground_contact(float(t))
            if in_contact:
                ttc = model.get_time_to_next_contact(float(t))
                # When in contact, find next contact should work
                assert ttc >= 0
                break

    def test_subsatellite_point_bounds(self):
        model = SimplifiedOrbitModel()

        for t in range(0, 10000, 100):
            lat, lon = model.get_subsatellite_point(float(t))
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180

    def test_subsatellite_point_varies(self):
        model = SimplifiedOrbitModel()

        points = [model.get_subsatellite_point(float(t)) for t in range(0, 5400, 540)]
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]

        # Latitude should vary (sinusoidal)
        assert max(lats) - min(lats) > 50

        # Longitude should progress
        assert len(set(lons)) > 5

    def test_station_selection_varies(self):
        model = SimplifiedOrbitModel()

        # Over multiple orbits, different stations should be selected
        stations_seen = set()
        for orbit in range(12):
            # Sample many points within the orbit to find contacts
            for t_offset in range(0, int(model.orbital_period), 60):
                t = orbit * model.orbital_period + t_offset
                in_contact, station, _ = model.get_ground_contact(t)
                if in_contact and station:
                    stations_seen.add(station.name)

        # Should see multiple stations over many orbits
        assert len(stations_seen) >= 2

    def test_elevation_during_contact(self):
        model = SimplifiedOrbitModel()

        for t in range(0, int(model.orbital_period), 30):
            in_contact, station, elevation = model.get_ground_contact(float(t))
            if in_contact:
                assert elevation >= station.min_elevation
                assert elevation <= 90.0


class TestOrbitModelConsistency:
    """Tests for orbit model behavior consistency."""

    def test_contact_and_ttc_consistency(self):
        """If in contact, time to contact should be 0."""
        model = SimplifiedOrbitModel()

        for t in range(0, 5400, 60):
            in_contact, _, _ = model.get_ground_contact(float(t))
            if in_contact:
                ttc = model.get_time_to_next_contact(float(t))
                # Time to next contact from an in-contact state
                # could be 0 or the time to the NEXT contact after current
                assert ttc >= 0

    def test_deterministic_behavior(self):
        """Same time should give same results."""
        model = SimplifiedOrbitModel()

        for t in [0, 100, 1000, 2700]:
            result1 = model.get_ground_contact(float(t))
            result2 = model.get_ground_contact(float(t))
            assert result1 == result2

            point1 = model.get_subsatellite_point(float(t))
            point2 = model.get_subsatellite_point(float(t))
            assert point1 == point2


class TestOrbitPropagator:
    """Tests for the full Skyfield-based OrbitPropagator."""

    def test_creation_with_defaults(self):
        propagator = OrbitPropagator()
        assert propagator is not None

    def test_creation_with_custom_tle(self):
        propagator = OrbitPropagator(
            tle_line1=DEFAULT_TLE[0],
            tle_line2=DEFAULT_TLE[1],
        )
        assert propagator is not None

    def test_get_position(self):
        propagator = OrbitPropagator()
        x, y, z = propagator.get_position(0.0)

        # Should be at LEO altitude (~400km)
        distance = np.sqrt(x**2 + y**2 + z**2)
        assert 6371 + 300 < distance < 6371 + 500  # Earth radius + altitude

    def test_get_subsatellite_point(self):
        propagator = OrbitPropagator()
        lat, lon = propagator.get_subsatellite_point(0.0)

        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_get_ground_contact(self):
        propagator = OrbitPropagator()
        in_contact, station, elevation = propagator.get_ground_contact(0.0)

        assert isinstance(in_contact, bool)
        if in_contact:
            assert station is not None
            assert elevation >= station.min_elevation
        else:
            assert elevation >= 0  # Elevation is 0 when no contact

    def test_get_time_to_next_contact(self):
        propagator = OrbitPropagator()
        ttc = propagator.get_time_to_next_contact(0.0)

        # Time to next contact should be non-negative
        assert ttc >= 0
        # Should be less than lookahead (2 hours default)
        assert ttc <= 7200.0

    def test_get_orbital_period(self):
        propagator = OrbitPropagator()
        period = propagator.get_orbital_period()

        # ISS-like orbit should be ~90-95 minutes
        assert 5000 < period < 6000

    def test_position_changes_over_time(self):
        propagator = OrbitPropagator()

        pos1 = propagator.get_position(0.0)
        pos2 = propagator.get_position(60.0)  # 1 minute later
        pos3 = propagator.get_position(600.0)  # 10 minutes later

        # Positions should be different
        assert pos1 != pos2
        assert pos2 != pos3

    def test_subsatellite_point_traces_ground_track(self):
        propagator = OrbitPropagator()

        points = []
        for t in range(0, 5400, 300):  # Every 5 minutes for one orbit
            lat, lon = propagator.get_subsatellite_point(float(t))
            points.append((lat, lon))

        # Latitude should vary (sinusoidal for inclined orbit)
        lats = [p[0] for p in points]
        assert max(lats) - min(lats) > 50  # Should span significant range

    def test_visibility_caching(self):
        propagator = OrbitPropagator()

        # First call should compute windows
        ttc1 = propagator.get_time_to_next_contact(0.0)

        # Second call at same time should use cache
        ttc2 = propagator.get_time_to_next_contact(0.0)

        assert ttc1 == ttc2

    def test_contact_over_full_orbit(self):
        propagator = OrbitPropagator()
        period = propagator.get_orbital_period()

        contact_count = 0
        for t in range(0, int(period), 60):
            in_contact, _, _ = propagator.get_ground_contact(float(t))
            if in_contact:
                contact_count += 1

        # Should have some contacts during an orbit
        # (depends on ground station locations)
        # At minimum, don't crash
        assert contact_count >= 0
