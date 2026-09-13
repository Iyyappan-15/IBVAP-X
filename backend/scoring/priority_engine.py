from typing import List, Tuple
from backend.config import settings
from backend.interfaces import ContextEvent, EventPriority, DirectionEnum, TimeContextEnum

class PriorityEngine:
    """
    Transparent rule-based scoring engine computing Event Priority Score (0-100).
    Note: These scoring weights are configurable prototype policy parameters designed to demonstrate
    explainable decision logic; they are not claimed as statistically validated threat probabilities.
    """

    def __init__(self):
        self.low_max = settings.PRIORITY_LOW_MAX
        self.medium_max = settings.PRIORITY_MEDIUM_MAX
        self.high_max = settings.PRIORITY_HIGH_MAX

    def compute_priority(
        self,
        context_event: ContextEvent,
        cross_camera_confirmed: bool = False,
        anomaly_score: float = None
    ) -> Tuple[EventPriority, float, List[str]]:
        """
        Computes EventPriority classification, total score (0-100), and structured reason strings.
        """
        score = 0.0
        reasons: List[str] = []

        # 1. Restricted Zone Entry (+45 base -> guarantees at least MEDIUM priority alert)
        if context_event.in_restricted_zone:
            score += 45.0
            z_label = context_event.zone_name or "Restricted Zone"
            reasons.append(f"Restricted zone entry detected ({z_label})")

        # 2. Night-Time Event (+20)
        if context_event.time_context == TimeContextEnum.NIGHT:
            score += 20.0
            reasons.append(f"Night-time operational context event ({context_event.time_context.value})")

        # 3. Loitering Above Threshold (+20)
        if context_event.loitering:
            score += 20.0
            reasons.append(f"Sustained loitering detected ({context_event.loitering_duration_seconds:.0f}s > threshold)")

        # 4. Movement Toward Boundary (+15)
        if context_event.direction == DirectionEnum.TOWARD_BOUNDARY:
            score += 15.0
            reasons.append(f"Trajectory movement vector directed toward border boundary ({context_event.direction.value})")

        # 5. Sensor Tampering & Direct Physical Assault (+35)
        if getattr(context_event, "tampering_detected", False):
            score += 35.0
            reasons.append("Physical sensor tampering / camera impact assault detected")

        # 6. Hostile Direct Frontal Approach (+25)
        if getattr(context_event, "hostile_approach", False):
            score += 25.0
            reasons.append("Rapid frontal approach toward perimeter surveillance post")

        # 7. Handheld Object / Weapon / Stone Detected (+20)
        if getattr(context_event, "holding_object", False):
            score += 20.0
            reasons.append("Suspicious handheld projectile / stone / weapon detected in target possession")

        # 8. Adverse Weather Cover (Fog / Low-Visibility Snow) (+15)
        if getattr(context_event, "adverse_weather", False):
            score += 15.0
            reasons.append("Adverse atmospheric cover (Fog / Low-visibility snow conditions)")

        # 9. Camera Physically Broken / Destroyed (+40) → VERY HIGH THREAT
        # Physical destruction of surveillance hardware is a severe escalation event.
        if getattr(context_event, "camera_broken", False):
            score += 40.0
            reasons.append("CRITICAL: Camera physically destroyed / lens shattered — sensor offline")

        # 10. Cross-Camera Confirmation (+15)
        if cross_camera_confirmed:
            score += 15.0
            reasons.append("Corroborating event confirmed across adjacent camera feed")

        # 11. Secondary Anomaly Signal (+10)
        if anomaly_score is not None and anomaly_score >= settings.ANOMALY_SCORE_THRESHOLD:
            score += 10.0
            reasons.append(f"Secondary anomaly engine signal detected (Anomaly Index: {anomaly_score:.2f})")

        # Cap total score at 100.0
        total_score = round(min(100.0, max(0.0, score)), 1)

        # Classification mapping
        if total_score <= self.low_max:
            priority = EventPriority.LOW
        elif total_score <= self.medium_max:
            priority = EventPriority.MEDIUM
        elif total_score <= self.high_max:
            priority = EventPriority.HIGH
        else:
            priority = EventPriority.CRITICAL

        return priority, total_score, reasons

    def compute_priority_breakdown(
        self,
        context_event: ContextEvent,
        cross_camera_confirmed: bool = False,
        anomaly_score: float = None
    ) -> List[dict]:
        """
        Returns an itemized breakdown of scoring factors with points and active status.
        """
        factors = [
            {
                "name": "Restricted Zone Entry",
                "points": 45.0,
                "active": bool(context_event.in_restricted_zone),
                "detail": f"Zone: {context_event.zone_name or 'Restricted Area'}" if context_event.in_restricted_zone else "Outside restricted boundary"
            },
            {
                "name": "Night-Time Operation Context",
                "points": 20.0,
                "active": context_event.time_context == TimeContextEnum.NIGHT,
                "detail": f"Context: {context_event.time_context.value}"
            },
            {
                "name": "Sustained Loitering",
                "points": 20.0,
                "active": bool(context_event.loitering),
                "detail": f"Duration: {context_event.loitering_duration_seconds:.0f}s (Threshold: {settings.LOITERING_THRESHOLD_SECONDS}s)" if context_event.loitering else "No prolonged loitering"
            },
            {
                "name": "Movement Toward Boundary",
                "points": 15.0,
                "active": context_event.direction == DirectionEnum.TOWARD_BOUNDARY,
                "detail": f"Vector: {context_event.direction.value}"
            },
            {
                "name": "Cross-Camera Corroboration",
                "points": 15.0,
                "active": bool(cross_camera_confirmed),
                "detail": "Corroborated across adjacent camera" if cross_camera_confirmed else "Single-camera observation"
            },
            {
                "name": "Secondary Anomaly Signal",
                "points": 10.0,
                "active": bool(anomaly_score is not None and anomaly_score >= settings.ANOMALY_SCORE_THRESHOLD),
                "detail": f"Anomaly score: {anomaly_score:.2f}" if anomaly_score is not None else "Normal kinematic motion"
            }
        ]

        # Append active situational threat factors
        if getattr(context_event, "tampering_detected", False):
            factors.append({
                "name": "Physical Sensor Tampering",
                "points": 35.0,
                "active": True,
                "detail": "Direct physical assault / lens impact"
            })
        if getattr(context_event, "hostile_approach", False):
            factors.append({
                "name": "Hostile Frontal Approach",
                "points": 25.0,
                "active": True,
                "detail": "Direct advance toward camera post"
            })
        if getattr(context_event, "holding_object", False):
            factors.append({
                "name": "Handheld Weapon / Stone",
                "points": 20.0,
                "active": True,
                "detail": "Suspicious object / projectile in possession"
            })
        if getattr(context_event, "adverse_weather", False):
            factors.append({
                "name": "Adverse Weather Cover (Fog / Snow)",
                "points": 15.0,
                "active": True,
                "detail": "Fog / low-visibility atmospheric occlusion"
            })
        if getattr(context_event, "camera_broken", False):
            factors.append({
                "name": "Camera Physically Destroyed",
                "points": 40.0,
                "active": True,
                "detail": "CRITICAL: Lens shattered / sensor offline — hardware attack confirmed"
            })

        return factors

