import pytest
import time
from backend.context.zones import ZoneManager
from backend.context.loitering import LoiteringDetector
from backend.context.time_context import TimeContextClassifier
from backend.context.direction import DirectionClassifier
from backend.interfaces import TimeContextEnum, DirectionEnum

def test_zone_manager_point_in_polygon():
    poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    zm = ZoneManager(zone_name="Restricted Zone 1", polygon_normalized=poly)

    assert zm.is_point_in_zone(0.5, 0.5) is True
    assert zm.is_point_in_zone(0.1, 0.1) is False

def test_loitering_detector_cooldown():
    detector = LoiteringDetector(threshold_seconds=5.0, cooldown_seconds=10.0)
    t0 = 1000.0

    # Entry at t0
    loitering, dur = detector.update_track_zone(track_id=1, zone_name="Restricted Zone 1", current_time=t0)
    assert loitering is False
    assert dur == 0.0

    # Check at t0 + 6 seconds (exceeds 5.0s threshold)
    loitering, dur = detector.update_track_zone(track_id=1, zone_name="Restricted Zone 1", current_time=t0 + 6.0)
    assert loitering is True
    assert dur == 6.0

    # Check immediately after at t0 + 7 seconds (cooldown in effect)
    loitering2, dur2 = detector.update_track_zone(track_id=1, zone_name="Restricted Zone 1", current_time=t0 + 7.0)
    assert loitering2 is False  # Cooldown prevents alert spam!
    assert dur2 == 7.0

def test_time_context_classifier():
    classifier = TimeContextClassifier(day_start_hour=6, night_start_hour=18)
    
    # 12:00 PM (1700000000 -> timestamp)
    import datetime
    dt_day = datetime.datetime(2026, 9, 11, 12, 0, 0).timestamp()
    dt_night = datetime.datetime(2026, 9, 11, 22, 0, 0).timestamp()

    assert classifier.classify(dt_day) == TimeContextEnum.DAY
    assert classifier.classify(dt_night) == TimeContextEnum.NIGHT

def test_direction_classifier_short_trajectory():
    classifier = DirectionClassifier(min_trajectory_frames=10)
    short_traj = [(10.0, 10.0), (12.0, 12.0)]
    assert classifier.classify_direction(short_traj) == DirectionEnum.UNCERTAIN

def test_direction_classifier_vectors():
    classifier = DirectionClassifier(min_trajectory_frames=3)
    
    # Toward boundary (0.0, -1.0) -> y decreases
    towards_traj = [(100.0, 300.0), (100.0, 200.0), (100.0, 100.0)]
    assert classifier.classify_direction(towards_traj, boundary_vector=(0.0, -1.0)) == DirectionEnum.TOWARD_BOUNDARY

    # Away from boundary -> y increases
    away_traj = [(100.0, 100.0), (100.0, 200.0), (100.0, 300.0)]
    assert classifier.classify_direction(away_traj, boundary_vector=(0.0, -1.0)) == DirectionEnum.AWAY
