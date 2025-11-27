import os
from typing import Optional


def get_eta_from_osrm(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    osrm_url: Optional[str] = None
) -> Optional[float]:
    """
    Get ETA from OSRM routing service.
    
    Args:
        start_lat: Start latitude
        start_lon: Start longitude
        end_lat: End latitude
        end_lon: End longitude
        osrm_url: OSRM service URL (defaults to environment variable or localhost)
    
    Returns:
        ETA in minutes or None if OSRM unavailable
    """
    import requests
    
    if osrm_url is None:
        osrm_url = os.getenv("OSRM_URL", "http://localhost:5000")
    
    try:
        url = f"{osrm_url}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            "overview": "false",
            "alternatives": "false",
            "steps": "false"
        }
        
        response = requests.get(url, params=params, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                duration_seconds = route.get("duration", 0)
                return duration_seconds / 60.0  # Convert to minutes
    except Exception as e:
        print(f"Warning: OSRM request failed: {e}. Using Haversine fallback.")
    
    return None

