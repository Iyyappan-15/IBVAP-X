from typing import List, Optional
from sqlalchemy.orm import Session
from backend.db.models import (
    CameraModel, EventModel, AlertModel, EvidenceModel, OperatorActionModel, EscalationModel, CameraHealthModel, UserModel
)
from backend.interfaces import (
    AlertOutput, CameraStatus, EventPriority, Actionability, AlertState, ReliabilityScore, UserRole
)

class Repository:
    def __init__(self, db: Session):
        self.db = db

    # User CRUD
    def get_user_by_username(self, username: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.username == username).first()

    def create_user(self, username: str, hashed_password: str, role: UserRole) -> UserModel:
        user = UserModel(username=username, hashed_password=hashed_password, role=role)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    # Camera CRUD
    def get_camera(self, camera_id: str) -> Optional[CameraModel]:
        return self.db.query(CameraModel).filter(CameraModel.camera_id == camera_id).first()

    def list_cameras(self) -> List[CameraModel]:
        return self.db.query(CameraModel).filter(CameraModel.is_active == True).all()

    def save_camera(self, camera_id: str, name: str, location: str, lat: float, lon: float, heading: float, fov: float, range_m: float) -> CameraModel:
        cam = self.get_camera(camera_id)
        if not cam:
            cam = CameraModel(camera_id=camera_id, name=name)
            self.db.add(cam)
        cam.name = name
        cam.location_description = location
        cam.latitude = lat
        cam.longitude = lon
        cam.heading_degrees = heading
        cam.fov_degrees = fov
        cam.range_meters = range_m
        self.db.commit()
        self.db.refresh(cam)
        return cam

    # Camera Health Records
    def save_camera_health(self, score: ReliabilityScore) -> CameraHealthModel:
        health = CameraHealthModel(
            camera_id=score.camera_id,
            timestamp=score.timestamp,
            blur_score=score.blur_score,
            brightness_score=score.brightness_score,
            frame_health_score=score.frame_health_score,
            obstruction_score=score.obstruction_score,
            composite_reliability_score=score.composite_reliability_score,
            status=score.status,
            reasons=score.reasons
        )
        self.db.add(health)
        
        # Update current camera status
        cam = self.get_camera(score.camera_id)
        if cam:
            cam.status = score.status
            
        self.db.commit()
        self.db.refresh(health)
        return health

    # Alert CRUD
    def save_alert(self, alert_data: AlertOutput) -> AlertModel:
        alert = AlertModel(
            alert_id=alert_data.alert_id,
            camera_id=alert_data.camera_id,
            track_id=alert_data.track_id,
            class_name=alert_data.class_name,
            timestamp=alert_data.timestamp,
            event_priority=alert_data.event_priority,
            event_priority_score=alert_data.event_priority_score,
            camera_reliability=alert_data.camera_reliability,
            camera_reliability_score=alert_data.camera_reliability_score,
            actionability=alert_data.actionability,
            action_recommendation=alert_data.action_recommendation,
            why_reasons=alert_data.why_reasons,
            state=alert_data.state,
            correlated_cameras=alert_data.correlated_cameras,
            anomaly_score=alert_data.anomaly_score
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert

    def get_alert(self, alert_id: str) -> Optional[AlertModel]:
        return self.db.query(AlertModel).filter(AlertModel.alert_id == alert_id).first()

    def list_alerts(self, limit: int = 50) -> List[AlertModel]:
        return self.db.query(AlertModel).order_by(AlertModel.created_at.desc()).limit(limit).all()

    def update_alert_state(self, alert_id: str, new_state: AlertState) -> Optional[AlertModel]:
        alert = self.get_alert(alert_id)
        if alert:
            alert.state = new_state
            self.db.commit()
            self.db.refresh(alert)
        return alert

    # Evidence CRUD
    def save_evidence(self, evidence_id: str, alert_id: str, event_id: str, camera_id: str, timestamp: float, video: str, snapshot: str, metadata: str, sha256: str) -> EvidenceModel:
        ev = EvidenceModel(
            evidence_id=evidence_id,
            alert_id=alert_id,
            event_id=event_id,
            camera_id=camera_id,
            timestamp=timestamp,
            video_filename=video,
            snapshot_filename=snapshot,
            metadata_filename=metadata,
            sha256_hash=sha256,
            is_sealed=True
        )
        self.db.add(ev)
        self.db.commit()
        self.db.refresh(ev)
        return ev

    def get_evidence(self, evidence_id: str) -> Optional[EvidenceModel]:
        return self.db.query(EvidenceModel).filter(EvidenceModel.evidence_id == evidence_id).first()
