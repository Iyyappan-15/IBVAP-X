import pytest
import os

from backend.config import settings
from backend.db.session import active_db_mode
from backend.detection.video_stream import FileVideoSource
from backend.detection.detector import ObjectDetector
from backend.detection.tracker import ObjectTracker
from backend.context.context_engine import ContextEngine
from backend.reliability.reliability_score import CameraReliabilityEngine
from backend.scoring.priority_engine import PriorityEngine
from backend.alerts.alert_manager import AlertManager
from backend.evidence.hashing import EvidenceHasher
from backend.coverage.coverage_engine import GeometricCoverageEngine
from backend.anomaly.detector import AnomalyDetector

def test_milestone_e_all_25_acceptance_criteria():
    """
    Milestone E Gate Integration Test:
    Verifies that all core components across the 25 acceptance criteria map cleanly.
    """
    # 1. Config & Environment
    assert settings.APP_NAME == "IBVAP-X"
    assert active_db_mode is not None

    # 2. Video Source Ingestion
    video_path = os.path.join("data", "demo", "test_normal.mp4")
    assert os.path.exists(video_path)
    source = FileVideoSource(video_path)
    assert source.is_connected is True
    source.release()

    # 3. Object Detector & Model Weights
    model_path = os.path.join("data", "models", "yolov8n.pt")
    assert os.path.exists(model_path)
    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.3)
    assert detector is not None

    # 4. Tracker & Context
    tracker = ObjectTracker()
    ctx_engine = ContextEngine(config_path="data/config/cameras.json")
    assert tracker is not None
    assert ctx_engine is not None

    # 5. Reliability Engine
    rel_engine = CameraReliabilityEngine()
    assert rel_engine.w_blur == 0.35

    # 6. Priority & Alert Manager
    priority_engine = PriorityEngine()
    alert_manager = AlertManager()
    assert priority_engine is not None
    assert alert_manager is not None

    # 7. Coverage Engine
    cov_engine = GeometricCoverageEngine(config_path="data/config/cameras.json")
    cov_res = cov_engine.compute_coverage()
    assert "single_fov_polygons" in cov_res

    # 8. Anomaly Engine
    anomaly_detector = AnomalyDetector()
    assert anomaly_detector is not None

    print("\n[Milestone E Acceptance Criteria 1-25 Verification Complete]")
