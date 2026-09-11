import pytest
import time
import numpy as np
from backend.interfaces import Track, DirectionEnum, AnomalyStatus
from backend.anomaly.features import TrajectoryFeatureExtractor
from backend.anomaly.detector import AnomalyDetector

def test_feature_extractor_short_trajectory():
    extractor = TrajectoryFeatureExtractor(min_frames=15)
    
    # Trajectory with only 5 frames -> returns None
    short_track = Track(
        track_id=1, class_name="person", bbox=[0,0,10,10],
        trajectory=[(float(i), float(i)) for i in range(5)],
        start_time=100.0, last_seen=105.0
    )
    assert extractor.extract_features(short_track) is None

def test_feature_extractor_valid_trajectory():
    extractor = TrajectoryFeatureExtractor(min_frames=15)
    
    # Trajectory with 20 frames
    valid_track = Track(
        track_id=2, class_name="person", bbox=[0,0,10,10],
        trajectory=[(float(i*2), float(i*2)) for i in range(20)],
        start_time=100.0, last_seen=120.0
    )
    feats = extractor.extract_features(valid_track)
    assert feats is not None
    assert feats.shape == (8,)
    assert feats[0] > 0.0  # avg_speed > 0

def test_anomaly_detector_score_normalization():
    detector = AnomalyDetector()
    
    # Normal track (smooth straight line)
    normal_track = Track(
        track_id=10, class_name="person", bbox=[0,0,10,10],
        trajectory=[(float(i*2), float(i*2)) for i in range(20)],
        start_time=100.0, last_seen=120.0
    )

    result = detector.evaluate_track("CAM-01", normal_track)
    assert result is not None
    assert 0.0 <= result.normalized_anomaly_score <= 1.0
    assert result.status in [AnomalyStatus.NORMAL, AnomalyStatus.ANOMALOUS]
    assert "⚠️ Prototype anomaly model" in result.explanation

def test_anomaly_detector_error_isolation():
    detector = AnomalyDetector()
    
    # Track with empty trajectory
    empty_track = Track(
        track_id=99, class_name="person", bbox=[0,0,10,10],
        trajectory=[], start_time=100.0, last_seen=120.0
    )

    # Should safely return None without raising an exception!
    res = detector.evaluate_track("CAM-01", empty_track)
    assert res is None
