import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False
    print("Warning: OR-Tools not available, using fallback optimization")


@dataclass
class AmbulanceCandidate:
    """Data class for ambulance candidate in optimization"""
    id: int
    vehicle_id: str
    lat: float
    lon: float
    is_available: bool
    equipment_level: str
    crew_size: int
    availability_score: float = 1.0


@dataclass
class DispatchSolution:
    """Data class for dispatch optimization result"""
    ambulance_id: int
    vehicle_id: str
    distance_km: float
    estimated_arrival_minutes: float
    score: float


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.
    
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def calculate_eta(distance_km: float, avg_speed_kmh: float = 60.0) -> float:
    """
    Calculate estimated time of arrival in minutes.
    
    Args:
        distance_km: Distance in kilometers
        avg_speed_kmh: Average ambulance speed in km/h (default 60 km/h)
    
    Returns:
        ETA in minutes
    """
    if distance_km <= 0:
        return 0.0
    time_hours = distance_km / avg_speed_kmh
    return time_hours * 60.0


def optimize_with_ortools(
    event_lat: float,
    event_lon: float,
    ambulances: List[AmbulanceCandidate],
    distance_weight: float = 0.7,
    availability_weight: float = 0.3
) -> Optional[DispatchSolution]:
    """
    Optimize ambulance dispatch using OR-Tools.
    
    Args:
        event_lat: Event latitude
        event_lon: Event longitude
        ambulances: List of available ambulance candidates
        distance_weight: Weight for distance in optimization (0-1)
        availability_weight: Weight for availability score (0-1)
    
    Returns:
        DispatchSolution or None if optimization fails
    """
    if not OR_TOOLS_AVAILABLE:
        return None
    
    if not ambulances:
        return None
    
    # Filter available ambulances
    available_ambulances = [a for a in ambulances if a.is_available]
    if not available_ambulances:
        return None
    
    # If only one ambulance, return it
    if len(available_ambulances) == 1:
        amb = available_ambulances[0]
        distance = haversine_distance(event_lat, event_lon, amb.lat, amb.lon)
        eta = calculate_eta(distance)
        return DispatchSolution(
            ambulance_id=amb.id,
            vehicle_id=amb.vehicle_id,
            distance_km=distance,
            estimated_arrival_minutes=eta,
            score=0.0
        )
    
    # Create distance matrix
    num_ambulances = len(available_ambulances)
    distance_matrix = []
    
    for amb in available_ambulances:
        distances = []
        # Distance from ambulance to event
        dist = haversine_distance(event_lat, event_lon, amb.lat, amb.lon)
        distances.append(int(dist * 1000))  # Convert to meters for OR-Tools
        distance_matrix.append(distances)
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(num_ambulances, 1, 0)  # 1 vehicle, start at depot 0
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        """Returns distance between two nodes"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][0]  # All go to event (index 0)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add availability penalty
    def availability_callback(from_index):
        """Returns availability penalty"""
        from_node = manager.IndexToNode(from_index)
        amb = available_ambulances[from_node]
        # Lower availability score = higher penalty
        penalty = int((1.0 - amb.availability_score) * 10000)
        return penalty
    
    availability_callback_index = routing.RegisterUnaryTransitCallback(availability_callback)
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        # Get selected ambulance
        index = routing.Start(0)
        selected_idx = manager.IndexToNode(index)
        selected_amb = available_ambulances[selected_idx]
        
        distance = haversine_distance(event_lat, event_lon, selected_amb.lat, selected_amb.lon)
        eta = calculate_eta(distance)
        
        # Calculate weighted score
        normalized_distance = distance / 100.0  # Normalize (assuming max 100km)
        score = (distance_weight * normalized_distance) + (availability_weight * (1.0 - selected_amb.availability_score))
        
        return DispatchSolution(
            ambulance_id=selected_amb.id,
            vehicle_id=selected_amb.vehicle_id,
            distance_km=distance,
            estimated_arrival_minutes=eta,
            score=score
        )
    
    return None


def optimize_fallback(
    event_lat: float,
    event_lon: float,
    ambulances: List[AmbulanceCandidate],
    distance_weight: float = 0.7,
    availability_weight: float = 0.3
) -> Optional[DispatchSolution]:
    """
    Fallback optimization using greedy approach when OR-Tools is unavailable.
    
    Args:
        event_lat: Event latitude
        event_lon: Event longitude
        ambulances: List of available ambulance candidates
        distance_weight: Weight for distance in optimization (0-1)
        availability_weight: Weight for availability score (0-1)
    
    Returns:
        DispatchSolution or None if no ambulances available
    """
    available_ambulances = [a for a in ambulances if a.is_available]
    if not available_ambulances:
        return None
    
    best_ambulance = None
    best_score = float('inf')
    
    for amb in available_ambulances:
        distance = haversine_distance(event_lat, event_lon, amb.lat, amb.lon)
        
        # Normalize distance (assuming max 100km)
        normalized_distance = min(distance / 100.0, 1.0)
        
        # Calculate weighted score (lower is better)
        score = (distance_weight * normalized_distance) + (availability_weight * (1.0 - amb.availability_score))
        
        if score < best_score:
            best_score = score
            best_ambulance = amb
            best_distance = distance
    
    if best_ambulance:
        eta = calculate_eta(best_distance)
        return DispatchSolution(
            ambulance_id=best_ambulance.id,
            vehicle_id=best_ambulance.vehicle_id,
            distance_km=best_distance,
            estimated_arrival_minutes=eta,
            score=best_score
        )
    
    return None


def optimize_dispatch(
    event_lat: float,
    event_lon: float,
    ambulances: List[AmbulanceCandidate],
    distance_weight: float = 0.7,
    availability_weight: float = 0.3
) -> Optional[DispatchSolution]:
    """
    Main optimization function that tries OR-Tools first, falls back to greedy.
    
    Args:
        event_lat: Event latitude
        event_lon: Event longitude
        ambulances: List of available ambulance candidates
        distance_weight: Weight for distance in optimization (0-1)
        availability_weight: Weight for availability score (0-1)
    
    Returns:
        DispatchSolution or None if optimization fails
    """
    if OR_TOOLS_AVAILABLE:
        result = optimize_with_ortools(event_lat, event_lon, ambulances, distance_weight, availability_weight)
        if result:
            return result
    
    # Fallback to greedy approach
    return optimize_fallback(event_lat, event_lon, ambulances, distance_weight, availability_weight)

