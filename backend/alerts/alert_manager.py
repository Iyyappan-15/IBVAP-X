import time
import uuid
import logging
from typing import Dict, List, Optional

from backend.config import settings
from backend.interfaces import (
    ContextEvent, ReliabilityScore, AlertOutput, EventPriority, AlertState
)
from backend.scoring.priority_engine import PriorityEngine
from backend.scoring.actionability import ActionabilityMatrix

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages alert creation, deduplication, state machine transitions, and memory/DB cache."""

    def __init__(self):
        self.priority_engine = PriorityEngine()
        self.actionability_matrix = ActionabilityMatrix()
        
        # Structure: alert_id -> AlertOutput
        self.active_alerts: Dict[str, AlertOutput] = {}
        # Cooldown map: (camera_id, track_id) -> last_alert_time
        self.alert_cooldowns: Dict[tuple, float] = {}

    def process_event(
        self,
        context_event: ContextEvent,
        reliability_score: ReliabilityScore,
        cross_camera_confirmed: bool = False,
        anomaly_score: float = None
    ) -> Optional[AlertOutput]:
        """Processes context event + reliability score and returns an AlertOutput if priority >= MEDIUM."""
        
        # Compute Priority
        priority, p_score, why_reasons = self.priority_engine.compute_priority(
            context_event=context_event,
            cross_camera_confirmed=cross_camera_confirmed,
            anomaly_score=anomaly_score
        )

        # Ignore LOW priority routine events unless anomaly detected
        if priority == EventPriority.LOW and not anomaly_score:
            return None

        # Check deduplication cooldown (same track on same camera)
        cooldown_key = (context_event.camera_id, context_event.track_id)
        last_time = self.alert_cooldowns.get(cooldown_key, 0.0)

        if (context_event.timestamp - last_time) < settings.LOITERING_COOLDOWN_SECONDS:
            # Duplicate alert in cooldown — update existing active alert if present
            for alert in self.active_alerts.values():
                if alert.camera_id == context_event.camera_id and alert.track_id == context_event.track_id:
                    alert.why_reasons = list(set(alert.why_reasons + why_reasons))
                    alert.event_priority_score = max(alert.event_priority_score, p_score)
                    return alert

        # Evaluate Actionability independently
        actionability, rec_text = self.actionability_matrix.evaluate(
            priority=priority,
            reliability_status=reliability_score.status
        )

        alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        alert = AlertOutput(
            alert_id=alert_id,
            camera_id=context_event.camera_id,
            track_id=context_event.track_id,
            class_name=context_event.class_name,
            timestamp=context_event.timestamp,
            event_priority=priority,
            event_priority_score=p_score,
            camera_reliability=reliability_score.status,
            camera_reliability_score=reliability_score.composite_reliability_score,
            actionability=actionability,
            action_recommendation=rec_text,
            why_reasons=why_reasons,
            state=AlertState.NEW,
            anomaly_score=anomaly_score
        )

        self.active_alerts[alert_id] = alert
        self.alert_cooldowns[cooldown_key] = context_event.timestamp
        logger.info(f"[AlertManager] Created alert '{alert_id}': Priority={priority.value}, Reliability={reliability_score.status.value}, Actionability={actionability.value}")

        return alert

    def get_alert(self, alert_id: str) -> Optional[AlertOutput]:
        return self.active_alerts.get(alert_id)

    def list_alerts(self, limit: int = 50) -> List[AlertOutput]:
        sorted_alerts = sorted(self.active_alerts.values(), key=lambda a: a.timestamp, reverse=True)
        return sorted_alerts[:limit]

    def update_state(self, alert_id: str, new_state: AlertState) -> Optional[AlertOutput]:
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.state = new_state
            logger.info(f"[AlertManager] Alert '{alert_id}' transition state -> {new_state.value}")
        return alert
