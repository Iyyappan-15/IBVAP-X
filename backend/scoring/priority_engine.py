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

        # 1. Restricted Zone Entry (+30)
        if context_event.in_restricted_zone:
            score += 30.0
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

        # 5. Cross-Camera Confirmation (+15)
        if cross_camera_confirmed:
            score += 15.0
            reasons.append("Corroborating event confirmed across adjacent camera feed")

        # 6. Secondary Anomaly Signal (+10)
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
