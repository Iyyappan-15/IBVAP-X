import os
import time
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AuditLogger:
    """Append-only audit logger recording system events and operator actions to file and DB."""

    def __init__(self, log_path: str = "data/evidence/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log_event(self, event_type: str, actor: str, target_id: str, details: dict):
        """Appends a new immutable audit record to file."""
        record = {
            "timestamp": time.time(),
            "datetime_str": datetime.now().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "target_id": target_id,
            "details": details
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"[AuditLog] {event_type} by {actor} on {target_id}")

    def read_audit_trail(self, limit: int = 50) -> list:
        """Reads recent audit records in reverse chronological order."""
        if not os.path.exists(self.log_path):
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        records.reverse()
        return records[:limit]
