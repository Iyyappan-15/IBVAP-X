import pytest
import time
import numpy as np

from backend.config import settings
from backend.interfaces import (
    Frame, Detection, Track, ContextEvent, ReliabilityScore, CameraStatus,
    EventPriority, Actionability, TimeContextEnum, DirectionEnum, AnalysisFrameResult, DetectionSource
)
from backend.detection.detector import ObjectDetector
from backend.reliability.reliability_score import CameraReliabilityEngine, apply_demo_degradation
from backend.alerts.alert_manager import AlertManager
from backend.scoring.priority_engine import PriorityEngine
from backend.scoring.actionability import ActionabilityMatrix
from backend.pipeline import IBVAPXPipeline


def test_model_class_reporting_transparency():
    """Test I & J: Model class reporting accurately reflects loaded weights and unsupported classes."""
    detector = ObjectDetector()
    info = detector.get_model_info()

    assert "model_name" in info
    assert info["model_name"] in ["yolov8n.pt", "ibvapx_custom.pt"]
    assert "supported_classes" in info
    assert "person" in info["supported_classes"]
    assert "unsupported_classes" in info
    assert "fence" in info["unsupported_classes"]
    assert "stone" in info["unsupported_classes"]


def test_alert_deduplication_same_track_no_multiplication():
    """Test C: Repeated same-frame event within cooldown updates active alert without returning new duplicate objects."""
    alert_manager = AlertManager()
    rel_score = ReliabilityScore(
        camera_id="CAM-01",
        timestamp=time.time(),
        blur_score=100.0, brightness_score=100.0, frame_health_score=100.0, obstruction_score=100.0,
        composite_reliability_score=100.0, status=CameraStatus.GOOD
    )
    ctx_event = ContextEvent(
        camera_id="CAM-01",
        track_id=42,
        class_name="person",
        timestamp=time.time(),
        in_restricted_zone=True,
        time_context=TimeContextEnum.NIGHT,
        direction=DirectionEnum.TOWARD_BOUNDARY
    )

    # Frame 1: First occurrence -> creates new alert
    alert1 = alert_manager.process_event(ctx_event, rel_score)
    assert alert1 is not None
    assert alert1.alert_id.startswith("ALT-")

    # Frame 2: Same event 0.1s later (within cooldown) -> must return None to prevent duplicate alert emission
    ctx_event2 = ctx_event.model_copy(update={"timestamp": ctx_event.timestamp + 0.1})
    alert2 = alert_manager.process_event(ctx_event2, rel_score)
    assert alert2 is None  # Deduplicated!

    # Active alerts list still retains exactly 1 alert
    active = alert_manager.list_alerts()
    assert len(active) == 1
    assert active[0].alert_id == alert1.alert_id


def test_rolling_camera_reliability_and_hysteresis():
    """Test E & F: Rolling window smoothing and temporal hysteresis on camera status transitions."""
    engine = CameraReliabilityEngine()
    now = time.time()

    # Feed 10 good frames -> status GOOD
    for i in range(10):
        img_good = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Add high variance texture
        img_good[:50, :50] = 255
        res = engine.calculate_reliability("CAM-TEST", i, now + i, img_good)

    assert res.status == CameraStatus.GOOD

    # Single noisy bad frame -> status must NOT jump immediately due to hysteresis
    bad_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_transient = engine.calculate_reliability("CAM-TEST", 11, now + 11, bad_img)
    assert res_transient.status == CameraStatus.GOOD  # Hysteresis guards against single-frame status flips

    # Feed 15 consecutive bad frames so rolling average drops into POOR/OFFLINE band
    for i in range(12, 28):
        res_bad = engine.calculate_reliability("CAM-TEST", i, now + i, bad_img)

    assert res_bad.status in [CameraStatus.POOR, CameraStatus.OFFLINE]


def test_camera_failure_does_not_inflate_event_priority():
    """Test G: Camera status (POOR/OFFLINE) does not inflate Event Priority threat score."""
    priority_engine = PriorityEngine()
    ctx_event = ContextEvent(
        camera_id="CAM-01",
        track_id=1,
        class_name="person",
        timestamp=time.time(),
        in_restricted_zone=False,
        time_context=TimeContextEnum.DAY,
        direction=DirectionEnum.LATERAL
    )

    p_level, p_score, reasons = priority_engine.compute_priority(ctx_event)
    assert p_level == EventPriority.LOW
    assert p_score <= 40.0
    assert not any("camera" in r.lower() for r in reasons)


def test_actionability_drops_on_camera_offline():
    """Test H: High priority event on an OFFLINE camera feed produces Actionability MEDIUM with camera verification advice."""
    act, rec_text = ActionabilityMatrix.evaluate(EventPriority.HIGH, CameraStatus.OFFLINE)
    assert act == Actionability.MEDIUM
    assert "verification" in rec_text.lower() or "offline" in rec_text.lower()


def test_pipeline_signature_backwards_compatibility():
    """Test J: IBVAPXPipeline process_frame returns Tuple[np.ndarray, List[AlertOutput]] and updates last_frame_result."""
    pipeline = IBVAPXPipeline()
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    frame_obj = Frame(camera_id="CAM-01", frame_id=1, timestamp=time.time())

    annotated, alerts = pipeline.process_frame(None, frame_obj, img)
    assert isinstance(annotated, np.ndarray)
    assert isinstance(alerts, list)

    # Check AnalysisFrameResult data contract
    assert pipeline.last_frame_result is not None
    assert isinstance(pipeline.last_frame_result, AnalysisFrameResult)
    assert pipeline.last_frame_result.frame_id == 1
    assert pipeline.last_frame_result.camera_id == "CAM-01"
