import time
import uuid
import logging
from typing import Dict, Optional, List

from backend.config import settings
from backend.interfaces import AlertOutput, AlertState, EventPriority
from backend.evidence.audit import AuditLogger

logger = logging.getLogger(__name__)

class EscalationManager:
    """Manages acknowledgement timeout timers and escalates unacknowledged high priority alerts."""

    def __init__(self, ack_timeout_seconds: float = None):
        self.ack_timeout_seconds = ack_timeout_seconds or settings.ACK_TIMEOUT_SECONDS
        self.audit_logger = AuditLogger()
        # Structure: alert_id -> escalation_level_str
        self.escalated_alerts: Dict[str, str] = {}

    def check_and_escalate_alerts(self, active_alerts: List[AlertOutput], current_time: float = None) -> List[AlertOutput]:
        """
        Scans active alerts. If an alert remains in NEW or ACTIVE state after timeout, escalates it to COMMAND_CENTRE.
        Returns list of newly escalated AlertOutput objects.
        """
        now = current_time or time.time()
        escalated_list: List[AlertOutput] = []

        for alert in active_alerts:
            # Only HIGH or CRITICAL priority alerts escalate on timeout
            if alert.event_priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
                if alert.state in [AlertState.NEW, AlertState.ACTIVE]:
                    elapsed = now - alert.timestamp

                    if elapsed >= self.ack_timeout_seconds:
                        if alert.alert_id not in self.escalated_alerts:
                            alert.state = AlertState.ESCALATED
                            self.escalated_alerts[alert.alert_id] = "COMMAND_CENTRE"
                            escalated_list.append(alert)

                            self.audit_logger.log_event(
                                event_type="ALERT_AUTO_ESCALATED",
                                actor="SYSTEM_TIMEOUT_WORKER",
                                target_id=alert.alert_id,
                                details={
                                    "elapsed_seconds": round(elapsed, 1),
                                    "timeout_threshold": self.ack_timeout_seconds,
                                    "escalated_to": "COMMAND_CENTRE"
                                }
                            )

                            logger.warning(
                                f"[Escalation] Alert '{alert.alert_id}' ESCALATED to COMMAND_CENTRE "
                                f"(Elapsed: {elapsed:.1f}s >= timeout {self.ack_timeout_seconds:.1f}s)"
                            )

        return escalated_list
