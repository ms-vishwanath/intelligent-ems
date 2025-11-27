import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from datetime import datetime
from typing import AsyncGenerator

# Database configuration from environment variables
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://pguser:pgpass123@62.72.32.151:5432/vectordb"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Alembic-ready base
Base = declarative_base()


class Ambulance(Base):
    """ORM model for ambulance resources"""
    __tablename__ = "ambulances"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String(50), unique=True, nullable=False, index=True)
    current_lat = Column(Float, nullable=False)
    current_lon = Column(Float, nullable=False)
    status = Column(String(20), default="available", nullable=False)  # available, dispatched, maintenance
    equipment_level = Column(String(20), default="basic", nullable=False)  # basic, advanced, critical
    crew_size = Column(Integer, default=2, nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Ambulance(id={self.id}, vehicle_id={self.vehicle_id}, status={self.status})>"


class Event(Base):
    """ORM model for emergency events"""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    patient_age = Column(Integer, nullable=False)
    patient_gender = Column(String(10), nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lon = Column(Float, nullable=False)
    reported_symptoms = Column(Text, nullable=False)
    caller_phone = Column(String(20), nullable=True)
    incident_type = Column(String(50), default="medical", nullable=False)
    predicted_severity = Column(Float, nullable=True)
    severity_category = Column(String(20), nullable=True)
    assigned_ambulance_id = Column(Integer, ForeignKey("ambulances.id"), nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending, dispatched, en_route, completed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Event(id={self.id}, severity={self.predicted_severity}, status={self.status})>"


class DispatchLog(Base):
    """ORM model for dispatch logging"""
    __tablename__ = "dispatch_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    ambulance_id = Column(Integer, ForeignKey("ambulances.id"), nullable=False)
    estimated_arrival_minutes = Column(Float, nullable=False)
    distance_km = Column(Float, nullable=False)
    severity_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<DispatchLog(event_id={self.event_id}, ambulance_id={self.ambulance_id})>"


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

