import pytest
import os
import cv2
import numpy as np

from backend.detection.video_stream import FileVideoSource
from backend.pipeline import IBVAPXPipeline

def test_pipeline_integration_execution():
    video_path = os.path.join("data", "demo", "test_normal.mp4")
    assert os.path.exists(video_path), "Test video must exist"

    source = FileVideoSource(file_path=video_path, camera_id="CAM-PIPE")
    pipeline = IBVAPXPipeline(enable_demo_degradation=False)

    frame_count = 0
    total_alerts = 0

    while source.is_connected and frame_count < 15:
        frame_obj = source.get_frame()
        if frame_obj is None:
            break

        frame_count += 1
        nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        annotated, alerts = pipeline.process_frame(source, frame_obj, img)
        assert annotated.shape == img.shape
        total_alerts += len(alerts)

    source.release()
    assert frame_count > 0, "Pipeline processed frames"
