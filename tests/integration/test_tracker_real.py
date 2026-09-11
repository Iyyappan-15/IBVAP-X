import pytest
import os
import cv2
import numpy as np

from backend.detection.detector import ObjectDetector
from backend.detection.tracker import ObjectTracker
from backend.interfaces import Frame, Track

def test_tracker_pipeline_integration():
    model_path = os.path.join("data", "models", "yolov8n.pt")
    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.3)
    tracker = ObjectTracker()

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8) + 100
    _, buf = cv2.imencode(".jpg", blank_img)
    frame_obj = Frame(camera_id="CAM-INTEG", frame_id=1, frame_bytes=buf.tobytes())

    detections = detector.detect(frame_obj, image_np=blank_img)
    tracks = tracker.update(detections, timestamp=frame_obj.timestamp)

    assert isinstance(tracks, list)
    annotated = tracker.draw_tracks_overlay(blank_img, tracks)
    assert annotated.shape == (480, 640, 3)
