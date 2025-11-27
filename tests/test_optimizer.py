import pytest
from app.optimizer import (
    optimize_dispatch,
    optimize_fallback,
    haversine_distance,
    calculate_eta,
    AmbulanceCandidate
)


def test_haversine_distance():
    """Test Haversine distance calculation"""
    # Distance between New York and Los Angeles (approximately 3944 km)
    ny_lat, ny_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    
    distance = haversine_distance(ny_lat, ny_lon, la_lat, la_lon)
    
    # Allow 5% tolerance
    assert 3700 < distance < 4200
    
    # Same point should be 0
    assert haversine_distance(ny_lat, ny_lon, ny_lat, ny_lon) == 0.0


def test_calculate_eta():
    """Test ETA calculation"""
    # 60 km at 60 km/h should be 60 minutes
    eta = calculate_eta(60.0, 60.0)
    assert abs(eta - 60.0) < 0.1
    
    # 30 km at 60 km/h should be 30 minutes
    eta = calculate_eta(30.0, 60.0)
    assert abs(eta - 30.0) < 0.1
    
    # Zero distance should be zero time
    assert calculate_eta(0.0) == 0.0


def test_optimize_fallback_single_ambulance():
    """Test fallback optimization with single ambulance"""
    ambulances = [
        AmbulanceCandidate(
            id=1,
            vehicle_id="AMB-001",
            lat=40.7128,
            lon=-74.0060,
            is_available=True,
            equipment_level="advanced",
            crew_size=2
        )
    ]
    
    event_lat, event_lon = 40.7580, -73.9855  # Close to ambulance
    
    result = optimize_fallback(event_lat, event_lon, ambulances)
    
    assert result is not None
    assert result.ambulance_id == 1
    assert result.vehicle_id == "AMB-001"
    assert result.distance_km > 0
    assert result.estimated_arrival_minutes > 0


def test_optimize_fallback_multiple_ambulances():
    """Test fallback optimization with multiple ambulances"""
    ambulances = [
        AmbulanceCandidate(
            id=1,
            vehicle_id="AMB-001",
            lat=40.7128,
            lon=-74.0060,
            is_available=True,
            equipment_level="advanced",
            crew_size=2
        ),
        AmbulanceCandidate(
            id=2,
            vehicle_id="AMB-002",
            lat=40.7580,  # Closer to event
            lon=-73.9855,
            is_available=True,
            equipment_level="basic",
            crew_size=2
        ),
        AmbulanceCandidate(
            id=3,
            vehicle_id="AMB-003",
            lat=40.7831,  # Farthest
            lon=-73.9712,
            is_available=True,
            equipment_level="advanced",
            crew_size=2
        )
    ]
    
    event_lat, event_lon = 40.7580, -73.9855  # Same as AMB-002
    
    result = optimize_fallback(event_lat, event_lon, ambulances)
    
    assert result is not None
    # Should select closest ambulance (AMB-002)
    assert result.ambulance_id == 2
    assert result.distance_km < 1.0  # Very close


def test_optimize_fallback_no_available_ambulances():
    """Test fallback optimization with no available ambulances"""
    ambulances = [
        AmbulanceCandidate(
            id=1,
            vehicle_id="AMB-001",
            lat=40.7128,
            lon=-74.0060,
            is_available=False,  # Not available
            equipment_level="advanced",
            crew_size=2
        )
    ]
    
    event_lat, event_lon = 40.7580, -73.9855
    
    result = optimize_fallback(event_lat, event_lon, ambulances)
    
    assert result is None


def test_optimize_dispatch():
    """Test main dispatch optimization function"""
    ambulances = [
        AmbulanceCandidate(
            id=1,
            vehicle_id="AMB-001",
            lat=40.7128,
            lon=-74.0060,
            is_available=True,
            equipment_level="advanced",
            crew_size=2
        ),
        AmbulanceCandidate(
            id=2,
            vehicle_id="AMB-002",
            lat=40.7580,
            lon=-73.9855,
            is_available=True,
            equipment_level="basic",
            crew_size=2
        )
    ]
    
    event_lat, event_lon = 40.7580, -73.9855
    
    result = optimize_dispatch(event_lat, event_lon, ambulances)
    
    assert result is not None
    assert result.ambulance_id in [1, 2]
    assert result.distance_km >= 0
    assert result.estimated_arrival_minutes >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

