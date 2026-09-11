import pytest
import time
from backend.interfaces import Track, ContextEvent, TimeContextEnum, DirectionEnum, EventPriority, CameraStatus
from backend.correlation.event_correlator import EventCorrelator
from backend.anomaly.detector import AnomalyDetector
from backend.coverage.coverage_engine import GeometricCoverageEngine
from backend.alerts.alert_manager import AlertManager
from backend.reliability.reliability_score import CameraReliabilityEngine

def test_milestone_d_advanced_intelligence_integration():
    """
    Milestone D Gate Integration Test:
    - Geometric coverage engine identifies modeled FOV tiers with mandatory disclaimer
    - Anomaly detector computes normalized anomaly score with prototype warning label
    - Cross-camera correlator matches events on adjacent cameras and enriches alert why_reasons list
    """
    now = time.time()

    # 1. Coverage Gap Mapping
    cov_engine = GeometricCoverageEngine(config_path="data/config/cameras.json")
    cov_result = cov_engine.compute_coverage()
    assert "coverage_tier_labels" in cov_result
    assert cov_result["coverage_tier_labels"]["none"] == "0 modeled camera FOVs"

    # 2. Anomaly Detection
    anomaly_detector = AnomalyDetector()
    erratic_track = Track(
        track_id=88, class_name="person", bbox=[10,10,50,50],
        trajectory=[(float(i*3), float((i**2) % 50)) for i in range(25)],  # Erratic zig-zag path
        start_time=now - 25.0, last_seen=now
    )

    anomaly_res = anomaly_detector.evaluate_track("CAM-01", erratic_track)
    assert anomaly_res is not None
    assert 0.0 <= anomaly_res.normalized_anomaly_score <= 1.0
    assert "⚠️ Prototype anomaly model" in anomaly_res.explanation

    # 3. Cross-Camera Correlation
    correlator = EventCorrelator(config_path="data/config/cameras.json")
    evt_cam1 = ContextEvent(
        camera_id="CAM-01", track_id=1, class_name="person", timestamp=now - 15.0,
        in_restricted_zone=True, time_context=TimeContextEnum.NIGHT, direction=DirectionEnum.TOWARD_BOUNDARY
    )
    evt_cam2 = ContextEvent(
        camera_id="CAM-02", track_id=2, class_name="person", timestamp=now,
        in_restricted_zone=True, time_context=TimeContextEnum.NIGHT, direction=DirectionEnum.TOWARD_BOUNDARY
    )

    _ = correlator.process_event(evt_cam1)
    corr_res = correlator.process_event(evt_cam2)
    assert corr_res is not None
    assert len(corr_res.camera_ids) >= 2

    # 4. Enriched Alert Output
    alert_mgr = AlertManager()
    rel_score = CameraReliabilityEngine().calculate_reliability("CAM-02", frame_id=1, timestamp=now)
    
    alert = alert_mgr.process_event(
        context_event=evt_cam2,
        reliability_score=rel_score,
        cross_camera_confirmed=True,
        anomaly_score=anomaly_res.normalized_anomaly_score
    )

    assert alert is not None
    assert alert.anomaly_score == anomaly_res.normalized_anomaly_score
    assert any("Corroborating event confirmed" in r for r in alert.why_reasons)

    print("\n[Milestone D Advanced Intelligence Verified]")
    print(f"  Coverage Tiers  : {cov_result['coverage_tier_labels']['single']} | {cov_result['coverage_tier_labels']['none']}")
    print(f"  Anomaly Result  : Score={anomaly_res.normalized_anomaly_score:.2f} ({anomaly_res.status.value})")
    print(f"  Cross-Camera    : {corr_res.explanation}")
