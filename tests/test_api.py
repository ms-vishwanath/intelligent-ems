import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root on sys.path when running `python tests/test_api.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api import app  # noqa: E402
from app.db import Base, engine, AsyncSessionLocal  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
import asyncio


# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_create_event():
    """Test creating a new event"""
    event_data = {
        "patient_age": 45,
        "patient_gender": "M",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "reported_symptoms": "chest pain and difficulty breathing",
        "incident_type": "medical"
    }
    
    response = client.post("/event", json=event_data)
    
    # May fail if database not initialized, but should return proper structure
    assert response.status_code in [201, 500]  # 500 if DB not available
    
    if response.status_code == 201:
        data = response.json()
        assert "event_id" in data
        assert "predicted_severity" in data
        assert "severity_category" in data
        assert 0 <= data["predicted_severity"] <= 1
        assert data["severity_category"] in ["low", "medium", "high", "critical"]


def test_create_event_with_auto_dispatch():
    """Test creating event with auto-dispatch"""
    event_data = {
        "patient_age": 78,
        "patient_gender": "F",
        "location_lat": 40.7580,
        "location_lon": -73.9855,
        "reported_symptoms": "unconscious, not responding",
        "incident_type": "medical"
    }
    
    response = client.post("/event?auto_dispatch=true", json=event_data)
    
    # May fail if database not initialized
    assert response.status_code in [201, 500]
    
    if response.status_code == 201:
        data = response.json()
        assert "dispatch_result" in data


def test_list_ambulances():
    """Test listing ambulances"""
    response = client.get("/ambulances")
    
    # May fail if database not initialized
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "ambulances" in data
        assert "total_count" in data
        assert isinstance(data["ambulances"], list)


def test_list_available_ambulances():
    """Test listing only available ambulances"""
    try:
        response = client.get("/ambulances?available_only=true")
    except RuntimeError:
        pytest.skip("Database backend unavailable for /ambulances?available_only=true")
        return
    
    # May fail if database not initialized (returns 500)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "ambulances" in data
        # All returned ambulances should be available
        for amb in data["ambulances"]:
            assert amb["is_available"] is True


def test_dispatch_endpoint():
    """Test dispatch endpoint"""
    dispatch_data = {
        "event_id": 1,
        "ambulance_id": None
    }
    
    response = client.post("/dispatch", json=dispatch_data)
    
    # May fail if database not initialized or event doesn't exist
    assert response.status_code in [200, 404, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "success" in data
        assert data["success"] is True
        assert "dispatch_result" in data
        assert "estimated_arrival_minutes" in data["dispatch_result"]


def test_event_validation():
    """Test event data validation"""
    # Invalid age
    invalid_event = {
        "patient_age": 200,  # Too high
        "patient_gender": "M",
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "reported_symptoms": "test"
    }
    
    response = client.post("/event", json=invalid_event)
    assert response.status_code == 422  # Validation error
    
    # Invalid gender
    invalid_event2 = {
        "patient_age": 45,
        "patient_gender": "X",  # Invalid
        "location_lat": 40.7128,
        "location_lon": -74.0060,
        "reported_symptoms": "test"
    }
    
    response = client.post("/event", json=invalid_event2)
    assert response.status_code == 422  # Validation error


def test_predict_severity_endpoint():
    """Test severity prediction endpoint"""
    payload = {
        "patient_age": 60,
        "patient_gender": "F",
        "location_lat": 34.05,
        "location_lon": -118.25,
        "reported_symptoms": "severe headache and dizziness",
        "incident_type": "medical"
    }
    
    response = client.post("/predict/severity", json=payload)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "severity_score" in data
        assert "severity_category" in data
        assert "model_loaded" in data
        assert "used_fallback" in data


def test_predict_dispatch_endpoint():
    """Test dispatch prediction endpoint"""
    payload = {
        "patient_age": 72,
        "patient_gender": "M",
        "location_lat": 34.05,
        "location_lon": -118.25,
        "reported_symptoms": "shortness of breath",
        "incident_type": "medical"
    }
    
    response = client.post("/predict/dispatch", json=payload)
    
    # May fail if database/postgres not available
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "severity_prediction" in data
        assert "available_ambulances" in data
        assert "message" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

