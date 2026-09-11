from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

# Enums
class CameraStatus(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    POOR = "POOR"
    OFFLINE = "OFFLINE"

class EventPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Actionability(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AlertState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"
    ESCALATED = "ESCALATED"

class TimeContextEnum(str, Enum):
    DAY = "DAY"
    NIGHT = "NIGHT"

class DirectionEnum(str, Enum):
    TOWARD_BOUNDARY = "TOWARD_BOUNDARY"
    AWAY = "AWAY"
    LATERAL = "LATERAL"
    UNCERTAIN = "UNCERTAIN"

class UserRole(str, Enum):
    BOP_OPERATOR = "BOP_OPERATOR"
    COMMAND_OPERATOR = "COMMAND_OPERATOR"
    ADMIN = "ADMIN"

class AnomalyStatus(str, Enum):
    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"

# Frame Schema
class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_id: str
    frame_id: int
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    frame_bytes: Optional[bytes] = None  # JPEG bytes or reference
    source_type: str = "file"  # file | webcam | rtsp | demo
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

# Detection Output Schema
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    camera_id: str
    timestamp: float
    frame_id: int

# Track Schema
class Track(BaseModel):
    track_id: int
    class_name: str
    bbox: List[float]
    trajectory: List[Tuple[float, float]]  # List of center (x, y) coordinates
    start_time: float
    last_seen: float
    estimated_speed: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)
    direction_enum: DirectionEnum = DirectionEnum.UNCERTAIN
    zone_history: List[str] = Field(default_factory=list)

# Context Event Schema
class ContextEvent(BaseModel):
    camera_id: str
    track_id: int
    class_name: str
    timestamp: float
    in_restricted_zone: bool = False
    zone_name: Optional[str] = None
    loitering: bool = False
    loitering_duration_seconds: float = 0.0
    time_context: TimeContextEnum = TimeContextEnum.DAY
    direction: DirectionEnum = DirectionEnum.UNCERTAIN

# Reliability Score Schema
class ReliabilityScore(BaseModel):
    camera_id: str
    timestamp: float
    blur_score: float  # 0 to 100
    brightness_score: float  # 0 to 100
    frame_health_score: float  # 0 to 100
    obstruction_score: float  # 0 to 100
    composite_reliability_score: float  # 0 to 100
    status: CameraStatus
    reasons: List[str] = Field(default_factory=list)

# Anomaly Detection Result Schema
class AnomalyResult(BaseModel):
    track_id: int
    camera_id: str
    timestamp: float
    raw_score: float
    normalized_anomaly_score: float  # 0.0 to 1.0
    status: AnomalyStatus
    is_anomalous: bool
    explanation: str

# Correlated Event Schema
class CorrelatedEvent(BaseModel):
    correlation_id: str
    camera_ids: List[str]
    event_type: str
    start_time: float
    end_time: float
    object_class: str
    explanation: str
    confirmed: bool = True

# Alert Output Schema (Signature 3-value decision model)
class AlertOutput(BaseModel):
    alert_id: str
    camera_id: str
    track_id: int
    class_name: str
    timestamp: float
    event_priority: EventPriority
    event_priority_score: float  # 0 to 100
    camera_reliability: CameraStatus
    camera_reliability_score: float  # 0 to 100
    actionability: Actionability
    action_recommendation: str
    why_reasons: List[str] = Field(default_factory=list)
    state: AlertState = AlertState.NEW
    evidence_ids: List[str] = Field(default_factory=list)
    correlated_cameras: List[str] = Field(default_factory=list)
    anomaly_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

# Evidence Record Schema
class EvidenceRecord(BaseModel):
    evidence_id: str
    alert_id: str
    event_id: str
    camera_id: str
    timestamp: float
    video_filename: str
    snapshot_filename: str
    metadata_filename: str
    sha256_hash: str
    is_sealed: bool = True
    created_at: datetime = Field(default_factory=datetime.now)

# Operator Action Schema
class OperatorAction(BaseModel):
    action_id: str
    alert_id: str
    operator_username: str
    action_type: AlertState  # ACKNOWLEDGED, REJECTED, UNCERTAIN
    timestamp: float
    notes: Optional[str] = None

# Coverage Result Schema
class CoverageResult(BaseModel):
    camera_id: str
    latitude: float
    longitude: float
    heading_degrees: float
    fov_degrees: float
    range_meters: float
    coverage_tier: str  # "2+ modeled camera FOVs" | "1 modeled camera FOV" | "0 modeled camera FOVs"
    confidence: str  # HIGH | MEDIUM | LOW
    is_blindspot: bool = False
