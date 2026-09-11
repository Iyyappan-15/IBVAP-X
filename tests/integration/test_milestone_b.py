import pytest
import time
from backend.interfaces import (
    ContextEvent, ReliabilityScore, CameraStatus, EventPriority, Actionability, TimeContextEnum, DirectionEnum
)
from backend.reliability.reliability_score import CameraReliabilityEngine, apply_demo_degradation
from backend.alerts.alert_manager import AlertManager
import numpy as np

def test_milestone_b_signature_card_verification():
    """
    Milestone B Gate Integration Test:
    Verifies that a high priority context event observed on a degraded camera feed produces:
    - Event Priority: HIGH or CRITICAL
    - Camera Reliability: DEGRADED or POOR
    - Actionability: MEDIUM
    - Three separate values maintained independently with structured reason breakdowns.
    """
    now = time.time()
    alert_manager = AlertManager()
    reliability_engine = CameraReliabilityEngine()

    # 1. Generate Context Event (Restricted zone + Night + Loitering + Toward Boundary -> HIGH/CRITICAL)
    ctx_event = ContextEvent(
        camera_id="CAM-03",
        track_id=17,
        class_name="person",
        timestamp=now,
        in_restricted_zone=True,
        zone_name="Restricted Waterway Delta",
        loitering=True,
        loitering_duration_seconds=43.0,
        time_context=TimeContextEnum.NIGHT,
        direction=DirectionEnum.TOWARD_BOUNDARY
    )

    # 2. Generate Degraded Camera Frame & Calculate Reliability Score
    sharp_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    degraded_image = apply_demo_degradation(sharp_image, blur_ksize=51, brightness_factor=0.2)
    
    reliability_score = reliability_engine.calculate_reliability(
        camera_id="CAM-03",
        frame_id=10,
        timestamp=now,
        image_np=degraded_image
    )

    # Verify reliability is DEGRADED or POOR
    assert reliability_score.status in [CameraStatus.DEGRADED, CameraStatus.POOR, CameraStatus.OFFLINE]

    # 3. Process through Alert Manager
    alert = alert_manager.process_event(
        context_event=ctx_event,
        reliability_score=reliability_score,
        cross_camera_confirmed=True
    )

    assert alert is not None
    
    # 4. VERIFY THE SIGNATURE IBVAP-X DECISION MODEL
    # Event Priority is HIGH/CRITICAL
    assert alert.event_priority in [EventPriority.HIGH, EventPriority.CRITICAL]
    
    # Camera Reliability is DEGRADED/POOR
    assert alert.camera_reliability in [CameraStatus.DEGRADED, CameraStatus.POOR]
    
    # Actionability is MEDIUM (NOT collapsed into HIGH - penalty = score)
    assert alert.actionability == Actionability.MEDIUM
    
    # Verify structured explanations list is populated
    assert len(alert.why_reasons) >= 4
    assert any("Restricted zone" in r for r in alert.why_reasons)
    assert any("Night-time" in r for r in alert.why_reasons)
    assert any("loitering" in r for r in alert.why_reasons)

    print("\n[Milestone B Signature Card Verified]")
    print(f"  Event Priority     : {alert.event_priority.value} ({alert.event_priority_score:.0f}/100)")
    print(f"  Camera Reliability : {alert.camera_reliability.value} ({alert.camera_reliability_score:.0f}%)")
    print(f"  Actionability      : {alert.actionability.value}")
    print(f"  Reasons            : {alert.why_reasons}")
