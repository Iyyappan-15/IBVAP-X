import pytest
import time
from backend.interfaces import (
    ContextEvent, ReliabilityScore, CameraStatus, EventPriority, Actionability, AlertState, TimeContextEnum, DirectionEnum
)
from backend.scoring.priority_engine import PriorityEngine
from backend.scoring.actionability import ActionabilityMatrix
from backend.scoring.explanations import ExplanationGenerator
from backend.alerts.alert_manager import AlertManager

def test_priority_engine_computation():
    engine = PriorityEngine()
    now = time.time()

    # Event with Restricted Zone + Night + Loitering + Toward Boundary -> High Priority
    ctx = ContextEvent(
        camera_id="CAM-03", track_id=17, class_name="person", timestamp=now,
        in_restricted_zone=True, zone_name="Restricted Delta",
        loitering=True, loitering_duration_seconds=45.0,
        time_context=TimeContextEnum.NIGHT,
        direction=DirectionEnum.TOWARD_BOUNDARY
    )

    priority, score, reasons = engine.compute_priority(ctx, cross_camera_confirmed=True)
    assert score == 100.0  # 30 + 20 + 20 + 15 + 15 = 100
    assert priority == EventPriority.CRITICAL
    assert len(reasons) == 5

def test_actionability_matrix():
    # HIGH Priority + DEGRADED Reliability -> Actionability MEDIUM
    act, text = ActionabilityMatrix.evaluate(EventPriority.HIGH, CameraStatus.DEGRADED)
    assert act == Actionability.MEDIUM
    assert "DEGRADED" in text

    # CRITICAL Priority + GOOD Reliability -> Actionability HIGH
    act_crit, _ = ActionabilityMatrix.evaluate(EventPriority.CRITICAL, CameraStatus.GOOD)
    assert act_crit == Actionability.HIGH

def test_alert_manager_dedup_and_state():
    manager = AlertManager()
    now = time.time()

    ctx = ContextEvent(
        camera_id="CAM-01", track_id=5, class_name="person", timestamp=now,
        in_restricted_zone=True, zone_name="Alpha",
        loitering=False, time_context=TimeContextEnum.NIGHT,
        direction=DirectionEnum.TOWARD_BOUNDARY
    )
    rel = ReliabilityScore(
        camera_id="CAM-01", timestamp=now, blur_score=90.0, brightness_score=90.0,
        frame_health_score=100.0, obstruction_score=90.0, composite_reliability_score=92.5,
        status=CameraStatus.GOOD, reasons=[]
    )

    alert1 = manager.process_event(ctx, rel)
    assert alert1 is not None
    assert alert1.state == AlertState.NEW

    # State update
    updated = manager.update_state(alert1.alert_id, AlertState.ACKNOWLEDGED)
    assert updated.state == AlertState.ACKNOWLEDGED
