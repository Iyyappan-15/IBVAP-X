from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship
from backend.interfaces import CameraStatus, EventPriority, Actionability, AlertState, UserRole

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.BOP_OPERATOR)
    created_at = Column(DateTime, default=datetime.utcnow)

class CameraModel(Base):
    __tablename__ = "cameras"

    camera_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    location_description = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading_degrees = Column(Float, nullable=True, default=0.0)
    fov_degrees = Column(Float, nullable=True, default=60.0)
    range_meters = Column(Float, nullable=True, default=150.0)
    status = Column(SQLEnum(CameraStatus), default=CameraStatus.GOOD)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    health_records = relationship("CameraHealthModel", back_populates="camera")
    events = relationship("EventModel", back_populates="camera")
    alerts = relationship("AlertModel", back_populates="camera")

class CameraHealthModel(Base):
    __tablename__ = "camera_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.camera_id"), nullable=False)
    timestamp = Column(Float, nullable=False, index=True)
    blur_score = Column(Float, nullable=False)
    brightness_score = Column(Float, nullable=False)
    frame_health_score = Column(Float, nullable=False)
    obstruction_score = Column(Float, nullable=False)
    composite_reliability_score = Column(Float, nullable=False)
    status = Column(SQLEnum(CameraStatus), nullable=False)
    reasons = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("CameraModel", back_populates="health_records")

class EventModel(Base):
    __tablename__ = "events"

    event_id = Column(String(100), primary_key=True)
    camera_id = Column(String(50), ForeignKey("cameras.camera_id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    timestamp = Column(Float, nullable=False, index=True)
    in_restricted_zone = Column(Boolean, default=False)
    zone_name = Column(String(50), nullable=True)
    loitering = Column(Boolean, default=False)
    loitering_duration_seconds = Column(Float, default=0.0)
    time_context = Column(String(20), nullable=True)
    direction = Column(String(30), nullable=True)
    anomaly_score = Column(Float, nullable=True)
    is_anomalous = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("CameraModel", back_populates="events")

class AlertModel(Base):
    __tablename__ = "alerts"

    alert_id = Column(String(100), primary_key=True)
    camera_id = Column(String(50), ForeignKey("cameras.camera_id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    timestamp = Column(Float, nullable=False, index=True)
    event_priority = Column(SQLEnum(EventPriority), nullable=False)
    event_priority_score = Column(Float, nullable=False)
    camera_reliability = Column(SQLEnum(CameraStatus), nullable=False)
    camera_reliability_score = Column(Float, nullable=False)
    actionability = Column(SQLEnum(Actionability), nullable=False)
    action_recommendation = Column(Text, nullable=False)
    why_reasons = Column(JSON, nullable=False)
    state = Column(SQLEnum(AlertState), default=AlertState.NEW, index=True)
    correlated_cameras = Column(JSON, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    sync_status = Column(String(20), default="SYNCED")  # SYNCED | PENDING_SYNC
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("CameraModel", back_populates="alerts")
    evidence_records = relationship("EvidenceModel", back_populates="alert")
    operator_actions = relationship("OperatorActionModel", back_populates="alert")
    escalations = relationship("EscalationModel", back_populates="alert")

class EvidenceModel(Base):
    __tablename__ = "evidence"

    evidence_id = Column(String(100), primary_key=True)
    alert_id = Column(String(100), ForeignKey("alerts.alert_id"), nullable=False)
    event_id = Column(String(100), nullable=False)
    camera_id = Column(String(50), nullable=False)
    timestamp = Column(Float, nullable=False)
    video_filename = Column(String(255), nullable=False)
    snapshot_filename = Column(String(255), nullable=False)
    metadata_filename = Column(String(255), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    is_sealed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("AlertModel", back_populates="evidence_records")

class OperatorActionModel(Base):
    __tablename__ = "operator_actions"

    action_id = Column(String(100), primary_key=True)
    alert_id = Column(String(100), ForeignKey("alerts.alert_id"), nullable=False)
    operator_username = Column(String(50), nullable=False)
    action_type = Column(SQLEnum(AlertState), nullable=False)
    timestamp = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("AlertModel", back_populates="operator_actions")

class EscalationModel(Base):
    __tablename__ = "escalations"

    escalation_id = Column(String(100), primary_key=True)
    alert_id = Column(String(100), ForeignKey("alerts.alert_id"), nullable=False)
    escalation_level = Column(String(50), nullable=False)  # BOP | COMMAND_CENTRE
    triggered_at = Column(Float, nullable=False)
    reason = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    alert = relationship("AlertModel", back_populates="escalations")
