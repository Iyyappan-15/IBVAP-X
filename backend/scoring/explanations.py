from datetime import datetime
from typing import List, Dict, Any
from backend.interfaces import AlertOutput

class ExplanationGenerator:
    """Constructs structured, human-readable explainability reports for operator alerts."""

    @staticmethod
    def generate_explanation(alert: AlertOutput) -> Dict[str, Any]:
        dt_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        reliability_warning = ""
        if alert.camera_reliability in ["DEGRADED", "POOR"]:
            reliability_warning = (
                f"⚠️ Observation reliability is {alert.camera_reliability.value} ({alert.camera_reliability_score:.0f}%). "
                f"Detection confidence is high, but camera condition is degraded. Verify camera feed before escalation."
            )

        return {
            "what": f"Detected object: {alert.class_name.upper()} (Track #{alert.track_id})",
            "where": f"Camera Location: {alert.camera_id}",
            "when": f"Timestamp: {dt_str}",
            "why_reasons": alert.why_reasons,
            "how_reliable": f"Camera Reliability: {alert.camera_reliability_score:.0f}% ({alert.camera_reliability.value})",
            "event_priority": f"{alert.event_priority.value} ({alert.event_priority_score:.0f}/100)",
            "actionability": alert.actionability.value,
            "action_recommendation": alert.action_recommendation,
            "reliability_warning": reliability_warning
        }
