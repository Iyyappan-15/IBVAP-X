import pytest
import time
from backend.detection.tracker import ObjectTracker, PureIoUTracker, calculate_iou
from backend.interfaces import Detection, Track

def test_calculate_iou():
    box1 = [10.0, 10.0, 50.0, 50.0]
    box2 = [10.0, 10.0, 50.0, 50.0]
    assert calculate_iou(box1, box2) == 1.0

    box3 = [100.0, 100.0, 200.0, 200.0]
    assert calculate_iou(box1, box3) == 0.0

def test_pure_iou_tracker_update():
    tracker = PureIoUTracker(max_stale_frames=5, iou_threshold=0.3)
    now = time.time()

    det1 = Detection(
        class_id=0, class_name="person", confidence=0.9,
        bbox=[100.0, 100.0, 150.0, 200.0], camera_id="CAM-01",
        timestamp=now, frame_id=1
    )

    tracks = tracker.update([det1], timestamp=now)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "person"
    assert len(tracks[0].trajectory) == 1

    # Frame 2: Move slightly
    det2 = Detection(
        class_id=0, class_name="person", confidence=0.9,
        bbox=[105.0, 102.0, 155.0, 202.0], camera_id="CAM-01",
        timestamp=now + 0.1, frame_id=2
    )

    tracks2 = tracker.update([det2], timestamp=now + 0.1)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1  # Track ID maintained!
    assert len(tracks2[0].trajectory) == 2

def test_object_tracker_wrapper():
    tracker = ObjectTracker(max_stale_frames=10)
    now = time.time()

    det = Detection(
        class_id=0, class_name="person", confidence=0.85,
        bbox=[50.0, 50.0, 100.0, 150.0], camera_id="CAM-01",
        timestamp=now, frame_id=1
    )

    tracks = tracker.update([det], timestamp=now)
    assert isinstance(tracks, list)
    assert len(tracks) >= 1
    assert isinstance(tracks[0], Track)
