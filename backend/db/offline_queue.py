import logging
from typing import List
from sqlalchemy.orm import Session
from backend.db.models import AlertModel
from backend.interfaces import AlertOutput

logger = logging.getLogger(__name__)

class OfflineQueueManager:
    """Manages SQLite offline event queuing when primary PostgreSQL database is unreachable."""

    @staticmethod
    def queue_alert_offline(db: Session, alert: AlertOutput):
        """Saves alert to SQLite buffer with PENDING_SYNC status."""
        orm_alert = AlertModel(
            alert_id=alert.alert_id,
            camera_id=alert.camera_id,
            track_id=alert.track_id,
            class_name=alert.class_name,
            timestamp=alert.timestamp,
            event_priority=alert.event_priority,
            event_priority_score=alert.event_priority_score,
            camera_reliability=alert.camera_reliability,
            camera_reliability_score=alert.camera_reliability_score,
            actionability=alert.actionability,
            action_recommendation=alert.action_recommendation,
            why_reasons=alert.why_reasons,
            state=alert.state,
            sync_status="PENDING_SYNC"
        )
        db.add(orm_alert)
        db.commit()
        logger.info(f"[OfflineQueue] Queued alert '{alert.alert_id}' with status PENDING_SYNC")

    @staticmethod
    def get_pending_sync_alerts(db: Session) -> List[AlertModel]:
        return db.query(AlertModel).filter(AlertModel.sync_status == "PENDING_SYNC").all()

    @staticmethod
    def mark_alerts_synced(db: Session, alert_ids: List[str]):
        db.query(AlertModel).filter(AlertModel.alert_id.in_(alert_ids)).update({"sync_status": "SYNCED"}, synchronize_session=False)
        db.commit()
        logger.info(f"[OfflineQueue] Flushed and marked {len(alert_ids)} alerts as SYNCED.")
