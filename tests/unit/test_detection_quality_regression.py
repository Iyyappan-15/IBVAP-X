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


def test_suppress_nested_subboxes_eradicates_torso_bottle():
    """Verifies that false handheld items (bottle/stone) covering a person's torso are suppressed."""
    from backend.detection.detector import suppress_nested_subboxes

    person_det = Detection(
        class_id=0, class_name="person", confidence=0.88,
        bbox=[100.0, 100.0, 200.0, 350.0],  # w=100, h=250, area=25000
        camera_id="CAM-01", timestamp=time.time(), frame_id=1
    )
    # A large torso box misdetected as bottle (area=6000, 24% of person area, 100% inside person)
    torso_bottle = Detection(
        class_id=39, class_name="bottle", confidence=0.75,
        bbox=[120.0, 160.0, 180.0, 260.0],  # w=60, h=100, area=6000
        camera_id="CAM-01", timestamp=time.time(), frame_id=1
    )
    # A genuine tiny held item at the hand (area=400, 1.6% of person area)
    hand_phone = Detection(
        class_id=67, class_name="cell phone", confidence=0.82,
        bbox=[90.0, 200.0, 110.0, 220.0],  # w=20, h=20, area=400
        camera_id="CAM-01", timestamp=time.time(), frame_id=1
    )

    cleaned = suppress_nested_subboxes([person_det, torso_bottle, hand_phone])
    class_names = [d.class_name for d in cleaned]

    assert "person" in class_names
    assert "bottle" not in class_names  # Torso bottle MUST be suppressed!
    assert "cell phone" in class_names  # Small handheld item is preserved


def test_refine_detection_classes_quadruped_vs_person():
    """Verifies that horizontal quadruped animals are classified as dog, while upright persons are preserved."""
    from backend.detection.detector import refine_detection_classes

    upright_person = Detection(
        class_id=0, class_name="person", confidence=0.85,
        bbox=[100.0, 100.0, 180.0, 320.0],  # w=80, h=220, AR=2.75 (tall)
        camera_id="CAM-01", timestamp=time.time(), frame_id=1
    )
    dog_misclassified_as_person = Detection(
        class_id=0, class_name="person", confidence=0.70,
        bbox=[300.0, 350.0, 390.0, 420.0],  # w=90, h=70, AR=0.78 (horizontal quadruped)
        camera_id="CAM-01", timestamp=time.time(), frame_id=1
    )

    refined = refine_detection_classes([upright_person, dog_misclassified_as_person], img_height=600, img_width=800)
    assert refined[0].class_name == "person"
    assert refined[1].class_name == "dog"
    assert refined[1].class_id == 16


def test_tracker_upright_person_immune_to_bottle_corruption():
    """Verifies that an upright walking person track is locked to 'person' and cannot be rebranded into bottle or stone."""
    from backend.detection.tracker import ObjectTracker

    tracker = ObjectTracker()
    track_data = {
        "track_id": 4,
        "class_name": "bottle",
        "bbox": [200.0, 150.0, 290.0, 380.0],  # w=90, h=230, AR=2.55 (upright person)
        "class_history": [("bottle", 0.91), ("bottle", 0.91), ("person", 0.85)],
        "label_stability": "LOW"
    }

    tracker._update_track_label_stability(track_data)
    assert track_data["class_name"] == "person"  # Guaranteed person protection!


def test_fence_detector_rejects_sky_fog_images():
    """Verifies that FenceDetector does not hallucinate fence in uniform foggy sky."""
    from backend.detection.fence_detector import FenceDetector

    fd = FenceDetector()
    # Foggy sky image: flat gradient with no diamond mesh
    fog_img = np.full((600, 800, 3), 200, dtype=np.uint8)
    # Add subtle horizontal fog bands
    fog_img[100:150, :] = 195
    fog_img[200:230, :] = 190

    results = fd.detect_fence(fog_img)
    assert len(results) == 0  # Zero false positive fence in sky!
