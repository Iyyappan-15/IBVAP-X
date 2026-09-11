import pytest
import time
from backend.interfaces import AlertOutput, EventPriority, CameraStatus, Actionability, AlertState
from backend.alerts.acknowledgement import OperatorAcknowledgementManager
from backend.alerts.escalation import EscalationManager

def test_operator_acknowledgement_record():
    manager = OperatorAcknowledgementManager()
    action = manager.record_action(
        alert_id="ALT-12345",
        operator_username="bop_operator",
        action_type=AlertState.ACKNOWLEDGED,
        notes="Verified intruder via thermal check"
    )

    assert action.alert_id == "ALT-12345"
    assert action.operator_username == "bop_operator"
    assert action.action_type == AlertState.ACKNOWLEDGED
    assert "Verified" in action.notes

def test_escalation_manager_timeout_trigger():
    escalation_mgr = EscalationManager(ack_timeout_seconds=5.0)  # Fast 5s timeout for test
    t0 = 1000.0

    alert = AlertOutput(
        alert_id="ALT-HIGH-1",
        camera_id="CAM-03",
        track_id=10,
        class_name="person",
        timestamp=t0,
        event_priority=EventPriority.HIGH,
        event_priority_score=75.0,
        camera_reliability=CameraStatus.GOOD,
        camera_reliability_score=90.0,
        actionability=Actionability.HIGH,
        action_recommendation="Immediate operator verification",
        why_reasons=["Restricted zone entry"],
        state=AlertState.NEW
    )

    # Check at t0 + 2s (before timeout)
    escalated_1 = escalation_mgr.check_and_escalate_alerts([alert], current_time=t0 + 2.0)
    assert len(escalated_1) == 0
    assert alert.state == AlertState.NEW

    # Check at t0 + 6s (after 5s timeout)
    escalated_2 = escalation_mgr.check_and_escalate_alerts([alert], current_time=t0 + 6.0)
    assert len(escalated_2) == 1
    assert alert.state == AlertState.ESCALATED
    assert alert.alert_id in escalation_mgr.escalated_alerts

def test_rejected_alert_does_not_escalate():
    escalation_mgr = EscalationManager(ack_timeout_seconds=5.0)
    t0 = 1000.0

    alert = AlertOutput(
        alert_id="ALT-REJ-1",
        camera_id="CAM-01",
        track_id=12,
        class_name="person",
        timestamp=t0,
        event_priority=EventPriority.HIGH,
        event_priority_score=75.0,
        camera_reliability=CameraStatus.GOOD,
        camera_reliability_score=90.0,
        actionability=Actionability.HIGH,
        action_recommendation="Immediate operator verification",
        why_reasons=["Restricted zone entry"],
        state=AlertState.REJECTED  # Operator REJECTED!
    )

    # Check after timeout
    escalated = escalation_mgr.check_and_escalate_alerts([alert], current_time=t0 + 10.0)
    assert len(escalated) == 0
    assert alert.state == AlertState.REJECTED  # Stays REJECTED, does NOT escalate!
