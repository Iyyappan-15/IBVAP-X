import pytest
import time
from backend.interfaces import ContextEvent, TimeContextEnum, DirectionEnum
from backend.correlation.camera_graph import CameraGraph
from backend.correlation.event_correlator import EventCorrelator

def test_camera_graph_adjacency():
    graph = CameraGraph(config_path="data/config/cameras.json")
    assert graph.is_adjacent("CAM-01", "CAM-02") is True
    assert graph.is_adjacent("CAM-01", "CAM-03") is False  # Non-adjacent

def test_event_correlator_time_window():
    correlator = EventCorrelator(config_path="data/config/cameras.json")
    now = time.time()

    evt1 = ContextEvent(
        camera_id="CAM-01", track_id=1, class_name="person", timestamp=now - 10.0,
        in_restricted_zone=True, time_context=TimeContextEnum.NIGHT, direction=DirectionEnum.TOWARD_BOUNDARY
    )
    
    evt2 = ContextEvent(
        camera_id="CAM-02", track_id=2, class_name="person", timestamp=now,
        in_restricted_zone=True, time_context=TimeContextEnum.NIGHT, direction=DirectionEnum.TOWARD_BOUNDARY
    )

    # First event on CAM-01
    res1 = correlator.process_event(evt1)
    assert res1 is None

    # Second event on adjacent CAM-02 within window -> triggers correlation!
    res2 = correlator.process_event(evt2)
    assert res2 is not None
    assert "CAM-01" in res2.camera_ids
    assert "CAM-02" in res2.camera_ids
    assert "not identity-confirmed" in res2.explanation
