from datetime import datetime
from typing import List, Dict, Any
from backend.interfaces import AlertOutput, CameraStatus

class ExplanationGenerator:
    """Constructs structured, human-readable explainability reports for operator alerts."""

    @staticmethod
    def generate_explanation(alert: AlertOutput) -> Dict[str, Any]:
        dt_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")

        rel_val = getattr(alert.camera_reliability, "value", str(alert.camera_reliability))
        prio_val = getattr(alert.event_priority, "value", str(alert.event_priority))
        act_val = getattr(alert.actionability, "value", str(alert.actionability))

        reliability_warning = ""
        if rel_val in ["DEGRADED", "POOR"]:
            reliability_warning = (
                f"⚠️ Observation reliability is {rel_val} ({alert.camera_reliability_score:.0f}%). "
                f"Detection confidence is high, but camera condition is degraded. Verify camera feed before physical escalation."
            )

        return {
            "what": f"Detected object: {alert.class_name.upper()} (Track #{alert.track_id})",
            "where": f"Camera Sector: {alert.camera_id}",
            "when": f"Timestamp: {dt_str}",
            "why_reasons": alert.why_reasons,
            "how_reliable": f"Camera Reliability: {alert.camera_reliability_score:.0f}% ({rel_val})",
            "event_priority": f"{prio_val} ({alert.event_priority_score:.0f}/100)",
            "actionability": act_val,
            "action_recommendation": alert.action_recommendation,
            "reliability_warning": reliability_warning,
            "evidence_reference": f"Evidence: {', '.join(alert.evidence_ids) if alert.evidence_ids else 'Sealed Snapshot & Clip Cached'}"
        }

