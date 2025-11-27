from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional, Tuple
from datetime import datetime
import os

from app.db import get_db, Ambulance, Event, DispatchLog, init_db
from app.schemas import (
    EventCreate, EventResponse, AmbulanceResponse, AmbulanceListResponse,
    DispatchRequest, DispatchResponse, DispatchResult,
    SeverityPredictionResponse, DispatchPredictionResponse
)
from app.model_service import ModelService
from app.optimizer import optimize_dispatch, AmbulanceCandidate, DispatchSolution
from app.utils import get_eta_from_osrm

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Emergency Response System",
    description="Predictive analytics and real-time medical resource optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model service (loaded on startup)
model_service: Optional[ModelService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_service
    
    # Initialize database
    await init_db()
    
    # Load ML model
    model_path = os.getenv("MODEL_PATH", "ml/model.pkl")
    model_service = ModelService(model_path)
    
    print("Intelligent EMS API started successfully")
    print(f"Model loaded: {model_service.is_loaded()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Intelligent EMS API")


@app.post("/event", response_model=EventResponse, status_code=201)
async def create_event(
    event_data: EventCreate,
    background_tasks: BackgroundTasks,
    auto_dispatch: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new emergency event and optionally dispatch an ambulance.
    
    - Predicts severity using ML model
    - Optionally triggers automatic dispatch
    - Returns event with predicted severity
    """
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    # Predict severity
    severity_score, severity_category, _ = model_service.predict_severity(event_data.dict())
    
    # Create event in database
    db_event = Event(
        patient_age=event_data.patient_age,
        patient_gender=event_data.patient_gender,
        location_lat=event_data.location_lat,
        location_lon=event_data.location_lon,
        reported_symptoms=event_data.reported_symptoms,
        caller_phone=event_data.caller_phone,
        incident_type=event_data.incident_type,
        predicted_severity=severity_score,
        severity_category=severity_category,
        status="pending"
    )
    
    db.add(db_event)
    await db.commit()
    await db.refresh(db_event)
    
    # Auto-dispatch if requested
    dispatch_result = None
    if auto_dispatch:
        dispatch_result = await _perform_dispatch(
            db_event,
            db
        )
        
        if dispatch_result:
            # Update event with assigned ambulance
            await db.execute(
                update(Event)
                .where(Event.id == db_event.id)
                .values(
                    assigned_ambulance_id=dispatch_result.ambulance_id,
                    status="dispatched"
                )
            )
            await db.commit()
            
            # Log dispatch in background
            background_tasks.add_task(
                _log_dispatch,
                db_event.id,
                dispatch_result,
                severity_score,
                db
            )
    
    return EventResponse(
        event_id=db_event.id,
        predicted_severity=severity_score,
        severity_category=severity_category,
        dispatch_result=dispatch_result,
        created_at=db_event.created_at
    )


@app.post("/predict/severity", response_model=SeverityPredictionResponse)
async def predict_severity_endpoint(event_data: EventCreate):
    """
    Predict severity for an event payload without persisting it.
    """
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    severity_score, severity_category, used_fallback = model_service.predict_severity(event_data.dict())
    
    return SeverityPredictionResponse(
        severity_score=severity_score,
        severity_category=severity_category,
        model_loaded=model_service.is_loaded(),
        used_fallback=used_fallback
    )


@app.post("/predict/dispatch", response_model=DispatchPredictionResponse)
async def predict_dispatch_endpoint(
    event_data: EventCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Predict severity and recommend an ambulance without creating a database record.
    """
    if model_service is None:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    severity_score, severity_category, used_fallback = model_service.predict_severity(event_data.dict())
    
    try:
        dispatch_result, available_count = await _select_dispatch_candidate(
            event_lat=event_data.location_lat,
            event_lon=event_data.location_lon,
            severity_score=severity_score,
            db=db
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate dispatch prediction: {exc}") from exc
    
    severity_response = SeverityPredictionResponse(
        severity_score=severity_score,
        severity_category=severity_category,
        model_loaded=model_service.is_loaded(),
        used_fallback=used_fallback
    )
    
    if dispatch_result:
        message = f"Recommended ambulance {dispatch_result.vehicle_id} with ETA {dispatch_result.estimated_arrival_minutes:.1f} minutes."
    else:
        message = "No available ambulances found for the provided location."
    
    return DispatchPredictionResponse(
        severity_prediction=severity_response,
        dispatch_result=dispatch_result,
        available_ambulances=available_count,
        message=message
    )


@app.get("/ambulances", response_model=AmbulanceListResponse)
async def list_ambulances(
    available_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    List all ambulances with their current status and location.
    
    - Returns all ambulances or only available ones
    - Includes location, status, and equipment information
    """
    query = select(Ambulance)
    
    if available_only:
        query = query.where(Ambulance.is_available == True)
    
    result = await db.execute(query)
    ambulances = result.scalars().all()
    
    ambulance_responses = [
        AmbulanceResponse(
            id=amb.id,
            vehicle_id=amb.vehicle_id,
            current_lat=amb.current_lat,
            current_lon=amb.current_lon,
            status=amb.status,
            equipment_level=amb.equipment_level,
            crew_size=amb.crew_size,
            is_available=amb.is_available
        )
        for amb in ambulances
    ]
    
    return AmbulanceListResponse(
        ambulances=ambulance_responses,
        total_count=len(ambulance_responses)
    )


@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch_ambulance(
    dispatch_request: DispatchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually trigger ambulance dispatch for an event.
    
    - Optimizes ambulance selection using OR-Tools
    - Falls back to greedy algorithm if OR-Tools unavailable
    - Returns dispatch result with ETA
    """
    # Get event
    event_result = await db.execute(
        select(Event).where(Event.id == dispatch_request.event_id)
    )
    event = event_result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Perform dispatch
    dispatch_result = await _perform_dispatch(
        event,
        db,
        preferred_ambulance_id=dispatch_request.ambulance_id
    )
    
    if not dispatch_result:
        raise HTTPException(
            status_code=404,
            detail="No available ambulances for dispatch"
        )
    
    # Update event
    await db.execute(
        update(Event)
        .where(Event.id == event.id)
        .values(
            assigned_ambulance_id=dispatch_result.ambulance_id,
            status="dispatched"
        )
    )
    await db.commit()
    
    # Log dispatch in background
    background_tasks.add_task(
        _log_dispatch,
        event.id,
        dispatch_result,
        event.predicted_severity or 0.5,
        db
    )
    
    return DispatchResponse(
        success=True,
        event_id=event.id,
        dispatch_result=dispatch_result,
        message=f"Ambulance {dispatch_result.vehicle_id} dispatched. ETA: {dispatch_result.estimated_arrival_minutes:.1f} minutes"
    )


async def _perform_dispatch(
    event: Event,
    db: AsyncSession,
    preferred_ambulance_id: Optional[int] = None
) -> Optional[DispatchResult]:
    """
    Internal function to perform ambulance dispatch optimization for persisted events.
    """
    dispatch_result, _ = await _select_dispatch_candidate(
        event_lat=event.location_lat,
        event_lon=event.location_lon,
        severity_score=event.predicted_severity or 0.5,
        db=db,
        preferred_ambulance_id=preferred_ambulance_id
    )
    return dispatch_result


async def _select_dispatch_candidate(
    event_lat: float,
    event_lon: float,
    severity_score: float,
    db: AsyncSession,
    preferred_ambulance_id: Optional[int] = None
) -> Tuple[Optional[DispatchResult], int]:
    """
    Shared helper to select the best ambulance candidate.
    
    Returns:
        Tuple of (DispatchResult or None, number of available ambulances considered)
    """
    query = select(Ambulance).where(Ambulance.is_available == True)
    result = await db.execute(query)
    ambulances = result.scalars().all()
    total_available = len(ambulances)
    
    if not ambulances:
        return None, total_available
    
    if preferred_ambulance_id:
        preferred = next((a for a in ambulances if a.id == preferred_ambulance_id), None)
        if preferred:
            ambulances = [preferred]
    
    candidates = [
        AmbulanceCandidate(
            id=amb.id,
            vehicle_id=amb.vehicle_id,
            lat=amb.current_lat,
            lon=amb.current_lon,
            is_available=amb.is_available,
            equipment_level=amb.equipment_level,
            crew_size=amb.crew_size,
            availability_score=1.0 if amb.status == "available" else 0.5
        )
        for amb in ambulances
    ]
    
    solution = optimize_dispatch(
        event_lat,
        event_lon,
        candidates,
        distance_weight=0.7,
        availability_weight=0.3
    )
    
    if not solution:
        return None, total_available
    
    selected_amb = next((a for a in ambulances if a.id == solution.ambulance_id), None)
    
    osrm_eta = None
    if selected_amb:
        osrm_eta = get_eta_from_osrm(
            selected_amb.current_lat,
            selected_amb.current_lon,
            event_lat,
            event_lon
        )
    
    final_eta = osrm_eta if osrm_eta is not None else solution.estimated_arrival_minutes
    
    dispatch_result = DispatchResult(
        ambulance_id=solution.ambulance_id,
        vehicle_id=solution.vehicle_id,
        estimated_arrival_minutes=final_eta,
        distance_km=solution.distance_km,
        severity_score=severity_score,
        confidence=0.85
    )
    
    return dispatch_result, total_available


async def _log_dispatch(
    event_id: int,
    dispatch_result: DispatchResult,
    severity_score: float,
    db: AsyncSession
):
    """
    Background task to log dispatch information.
    """
    try:
        log_entry = DispatchLog(
            event_id=event_id,
            ambulance_id=dispatch_result.ambulance_id,
            estimated_arrival_minutes=dispatch_result.estimated_arrival_minutes,
            distance_km=dispatch_result.distance_km,
            severity_score=severity_score
        )
        db.add(log_entry)
        await db.commit()
    except Exception as e:
        print(f"Error logging dispatch: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded() if model_service else False,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Emergency Response System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

