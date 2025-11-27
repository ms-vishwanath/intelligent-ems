from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class EventCreate(BaseModel):
    """Schema for creating a new emergency event"""
    patient_age: int = Field(..., ge=0, le=150, description="Patient age")
    patient_gender: str = Field(..., pattern="^(M|F|Other)$", description="Patient gender")
    location_lat: float = Field(..., ge=-90, le=90, description="Latitude of emergency location")
    location_lon: float = Field(..., ge=-180, le=180, description="Longitude of emergency location")
    reported_symptoms: str = Field(..., min_length=1, description="Reported symptoms")
    caller_phone: Optional[str] = Field(None, description="Caller phone number")
    incident_type: str = Field(default="medical", description="Type of incident")


class AmbulanceResponse(BaseModel):
    """Schema for ambulance resource response"""
    id: int
    vehicle_id: str
    current_lat: float
    current_lon: float
    status: str
    equipment_level: str
    crew_size: int
    is_available: bool


class DispatchResult(BaseModel):
    """Schema for dispatch optimization result"""
    ambulance_id: int
    vehicle_id: str
    estimated_arrival_minutes: float
    distance_km: float
    severity_score: float
    confidence: float


class EventResponse(BaseModel):
    """Schema for event response with predictions"""
    event_id: int
    predicted_severity: float = Field(..., ge=0, le=1, description="Predicted severity score (0-1)")
    severity_category: str = Field(..., description="Severity category: low, medium, high, critical")
    dispatch_result: Optional[DispatchResult] = None
    created_at: datetime


class AmbulanceListResponse(BaseModel):
    """Schema for listing ambulances"""
    ambulances: List[AmbulanceResponse]
    total_count: int


class DispatchRequest(BaseModel):
    """Schema for manual dispatch request"""
    event_id: int
    ambulance_id: Optional[int] = None  # If None, optimizer will choose


class DispatchResponse(BaseModel):
    """Schema for dispatch response"""
    success: bool
    event_id: int
    dispatch_result: DispatchResult
    message: str


class SeverityPredictionResponse(BaseModel):
    """Schema for standalone severity prediction responses"""
    severity_score: float = Field(..., ge=0, le=1, description="Predicted severity score (0-1)")
    severity_category: str = Field(..., description="Severity category: low, medium, high, critical")
    model_loaded: bool = Field(..., description="Indicates if trained model was used")
    used_fallback: bool = Field(..., description="True if heuristic fallback was used")


class DispatchPredictionResponse(BaseModel):
    """Schema for dispatch prediction without persisting events"""
    severity_prediction: SeverityPredictionResponse
    dispatch_result: Optional[DispatchResult] = None
    available_ambulances: int = Field(..., ge=0, description="Number of ambulances considered")
    message: str

