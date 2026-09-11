import time
import uuid
import logging
from typing import Optional

from backend.interfaces import OperatorAction, AlertState
from backend.evidence.audit import AuditLogger

logger = logging.getLogger(__name__)

class OperatorAcknowledgementManager:
    """Manages operator review actions (CONFIRM / REJECT / UNCERTAIN) and updates alert state."""

    def __init__(self):
        self.audit_logger = AuditLogger()

    def record_action(
        self,
        alert_id: str,
        operator_username: str,
        action_type: AlertState,
        notes: Optional[str] = None
    ) -> OperatorAction:
        """
        Records human verification action and writes append-only audit record.
        """
        action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
        now = time.time()

        action = OperatorAction(
            action_id=action_id,
            alert_id=alert_id,
            operator_username=operator_username,
            action_type=action_type,
            timestamp=now,
            notes=notes
        )

        self.audit_logger.log_event(
            event_type=f"OPERATOR_ACTION_{action_type.value}",
            actor=operator_username,
            target_id=alert_id,
            details={"action_id": action_id, "notes": notes or ""}
        )

        logger.info(f"[Acknowledgement] Alert '{alert_id}' updated to state {action_type.value} by operator '{operator_username}'")
        return action
