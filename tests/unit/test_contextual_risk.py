import time
import numpy as np
import pytest
from backend.interfaces import Track, EventPriority, CameraStatus, ReliabilityScore
from backend.context.context_engine import ContextEngine
from backend.scoring.priority_engine import PriorityEngine

def test_benign_normal_video_produces_low_risk():
    """
    Verifies that a normal domestic/uploaded video with a person standing or walking normally
    in front of a good-quality camera produces LOW priority (score <= 25.0), and NOT a false
    Critical (100/100) alarm.
    """
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    
    # Daytime timestamp (e.g. 14:00)
    day_ts = 1773568800.0  # ~2 PM UTC
    
    # 1. Normal camera reliability (Clear video, 95% sharpness, 0% obstruction)
    normal_rel = ReliabilityScore(
        camera_id="CAM-UPLOAD-01",
        timestamp=day_ts,
        blur_score=95.0,
        brightness_score=85.0,
        frame_health_score=100.0,
        obstruction_score=100.0,
        composite_reliability_score=96.0,
        status=CameraStatus.GOOD,
        reasons=[]
    )
    
    # 2. Normal person track standing/moving gently in room (height ~ 45% frame height)
    normal_person = Track(
        track_id=1,
        class_name="person",
        bbox=[200.0, 100.0, 320.0, 320.0],  # h = 220, ~45% of 480
        confidence=0.92,
        start_time=day_ts - 5.0,
        last_seen=day_ts,
        trajectory=[(260.0, 210.0), (261.0, 211.0), (260.0, 210.0), (262.0, 212.0)]
    )
    normal_person.initial_bbox = [200.0, 100.0, 320.0, 320.0]
    
    # 3. Analyze context with realistic image
    np.random.seed(42)
    dummy_frame = np.random.randint(60, 200, (480, 640, 3), dtype=np.uint8)
    ctx_event = context_engine.analyze_track(
        camera_id="CAM-UPLOAD-01",
        track=normal_person,
        frame_width=640,
        frame_height=480,
        timestamp=day_ts,
        image_np=dummy_frame,
        active_tracks=[normal_person],
        reliability_score=normal_rel
    )
    
    # Verify no false flags
    assert ctx_event.in_restricted_zone is False, "Uploaded video without configured zone must be outside restricted zone"
    assert ctx_event.hostile_approach is False, "Stationary/normal person must not be flagged as hostile approach"
    assert ctx_event.tampering_detected is False, "Good quality camera must not be flagged as tampered"
    assert ctx_event.camera_broken is False
    assert ctx_event.holding_object is False
    
    # 4. Compute priority
    priority, score, reasons = priority_engine.compute_priority(ctx_event)
    
    assert priority == EventPriority.LOW, f"Normal video must produce LOW priority, got {priority.value}"
    assert score <= 25.0, f"Expected score <= 25.0, got {score}"

def test_cylinder_object_produces_low_risk():
    """
    Verifies that a gas cylinder or bottle detected in the scene does not trigger
    hostile approach or tampering false alarms.
    """
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    day_ts = 1773568800.0
    
    cylinder_track = Track(
        track_id=2,
        class_name="gas cylinder",
        bbox=[340.0, 250.0, 420.0, 430.0],
        confidence=0.88,
        start_time=day_ts - 5.0,
        last_seen=day_ts,
        trajectory=[(380.0, 340.0), (380.0, 340.0), (380.0, 340.0)]
    )
    cylinder_track.initial_bbox = [340.0, 250.0, 420.0, 430.0]
    
    ctx_event = context_engine.analyze_track(
        camera_id="CAM-UPLOAD-01",
        track=cylinder_track,
        frame_width=640,
        frame_height=480,
        timestamp=day_ts
    )
    
    assert ctx_event.hostile_approach is False
    assert ctx_event.tampering_detected is False
    
    priority, score, reasons = priority_engine.compute_priority(ctx_event)
    assert priority == EventPriority.LOW
    assert score <= 25.0

def test_genuine_threat_escalation():
    """
    Verifies that genuine security threats (e.g., restricted zone intrusion + weapon)
    escalate correctly to HIGH or CRITICAL.
    """
    context_engine = ContextEngine()
    priority_engine = PriorityEngine()
    
    # CAM-01 has a configured restricted zone: [0.2, 0.2] to [0.8, 0.8]
    # Center (0.5, 0.56) is inside CAM-01's zone
    threat_track = Track(
        track_id=5,
        class_name="person",
        bbox=[280.0, 200.0, 360.0, 340.0],  # center ~ (320, 270) -> (0.5, 0.56) in 640x480
        confidence=0.95,
        start_time=1000.0,
        last_seen=1040.0,
        trajectory=[(320.0, 300.0), (320.0, 290.0), (320.0, 270.0)]
    )
    
    weapon_track = Track(
        track_id=6,
        class_name="knife",
        bbox=[310.0, 250.0, 340.0, 280.0],
        confidence=0.90,
        start_time=1000.0,
        last_seen=1040.0,
        trajectory=[(325.0, 265.0)]
    )
    
    ctx_event = context_engine.analyze_track(
        camera_id="CAM-01",
        track=threat_track,
        frame_width=640,
        frame_height=480,
        timestamp=1040.0,
        active_tracks=[threat_track, weapon_track]
    )
    
    assert ctx_event.in_restricted_zone is True, "Center (0.5, 0.56) must be inside CAM-01 restricted polygon"
    assert ctx_event.holding_object is True, "Weapon held adjacent to person must be detected"
    
    priority, score, reasons = priority_engine.compute_priority(ctx_event)
    assert priority in [EventPriority.HIGH, EventPriority.CRITICAL]
    assert score >= 65.0
