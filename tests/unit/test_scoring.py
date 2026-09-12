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

def test_priority_engine_factor_breakdown():
    engine = PriorityEngine()
    now = time.time()

    ctx = ContextEvent(
        camera_id="CAM-01", track_id=2, class_name="person", timestamp=now,
        in_restricted_zone=True, zone_name="Restricted Alpha",
        loitering=True, loitering_duration_seconds=25.0,
        time_context=TimeContextEnum.NIGHT,
        direction=DirectionEnum.TOWARD_BOUNDARY
    )

    factors = engine.compute_priority_breakdown(ctx, cross_camera_confirmed=False, anomaly_score=0.85)
    assert len(factors) == 6
    zone_f = next(f for f in factors if f["name"] == "Restricted Zone Entry")
    assert zone_f["active"] is True
    assert zone_f["points"] == 45.0

    night_f = next(f for f in factors if f["name"] == "Night-Time Operation Context")
    assert night_f["active"] is True
    assert night_f["points"] == 20.0

    anom_f = next(f for f in factors if f["name"] == "Secondary Anomaly Signal")
    assert anom_f["active"] is True
    assert anom_f["points"] == 10.0

def test_explanation_generator():
    now = time.time()
    from backend.interfaces import AlertOutput
    alert = AlertOutput(
        alert_id="ALT-101",
        camera_id="CAM-01",
        track_id=1,
        class_name="person",
        timestamp=now,
        event_priority=EventPriority.HIGH,
        event_priority_score=80.0,
        camera_reliability=CameraStatus.GOOD,
        camera_reliability_score=94.0,
        actionability=Actionability.HIGH,
        action_recommendation="Dispatch perimeter patrol.",
        why_reasons=["Restricted zone entry", "Night operation"]
    )
    expl = ExplanationGenerator.generate_explanation(alert)
    assert "what" in expl
    assert "PERSON" in expl["what"]
    assert "where" in expl
    assert "why_reasons" in expl
    assert "how_reliable" in expl
    assert "actionability" in expl
    assert expl["actionability"] == "HIGH"

