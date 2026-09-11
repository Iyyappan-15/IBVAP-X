import pytest
import os
from backend.detection.video_stream import (
    FileVideoSource, DemoVideoSource, WebcamVideoSource, RTSPVideoSource, VideoSourceError
)
from backend.interfaces import Frame

def test_file_video_source_valid():
    video_path = os.path.join("data", "demo", "test_normal.mp4")
    assert os.path.exists(video_path), "Synthetic test video must exist"

    source = FileVideoSource(file_path=video_path, camera_id="CAM-TEST")
    assert source.is_connected is True
    assert source.total_frames == 90

    frame = source.get_frame()
    assert frame is not None
    assert isinstance(frame, Frame)
    assert frame.camera_id == "CAM-TEST"
    assert frame.frame_id == 1
    assert frame.source_type == "file"
    assert "width" in frame.source_metadata

    source.release()
    assert source.is_connected is False

def test_file_video_source_missing():
    with pytest.raises(VideoSourceError):
        FileVideoSource(file_path="non_existent_video_path.mp4")

def test_demo_video_source():
    video_path = os.path.join("data", "demo", "test_normal.mp4")
    source = DemoVideoSource(file_path=video_path, camera_id="CAM-01", demo_name="Test Patrol")
    
    frame = source.get_frame()
    assert frame is not None
    assert frame.source_type == "demo"
    assert frame.source_metadata["demo_name"] == "Test Patrol"
    
    source.release()
