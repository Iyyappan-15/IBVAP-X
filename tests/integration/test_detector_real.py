import pytest
import os
import cv2
import numpy as np

from backend.detection.detector import ObjectDetector
from backend.interfaces import Frame, Detection

def test_detector_real_image_inference():
    model_path = os.path.join("data", "models", "yolov8n.pt")
    assert os.path.exists(model_path), "Model weights must exist for integration test"

    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.25)

    # Generate a realistic test image with a person-shaped pattern or real image
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 128
    
    # Run detector on frame
    _, buf = cv2.imencode(".jpg", img)
    frame_obj = Frame(camera_id="CAM-INTEG", frame_id=1, frame_bytes=buf.tobytes())
    
    detections = detector.detect(frame_obj)
    assert isinstance(detections, list)
    # Verify Detection schema attributes if any detection returned
    for det in detections:
        assert isinstance(det, Detection)
        assert isinstance(det.class_name, str)
        assert isinstance(det.confidence, float)
        assert len(det.bbox) == 4
