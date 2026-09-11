import pytest
import os
import numpy as np

from backend.detection.detector import ObjectDetector, ModelNotFoundError
from backend.interfaces import Frame, Detection

def test_missing_model_raises_error():
    with pytest.raises(ModelNotFoundError):
        ObjectDetector(model_path="data/models/non_existent_weights.pt")

def test_detector_initialization():
    model_path = os.path.join("data", "models", "yolov8n.pt")
    assert os.path.exists(model_path), "Model weights must exist for test"

    detector = ObjectDetector(
        model_path=model_path,
        confidence_threshold=0.5,
        target_classes=["person", "car"]
    )
    assert detector is not None
    assert detector.confidence_threshold == 0.5
    assert len(detector.target_classes) == 2

def test_detector_detect_shape():
    model_path = os.path.join("data", "models", "yolov8n.pt")
    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.3)

    # Blank frame for unit schema shape test
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2
    _, buf = cv2.imencode(".jpg", blank_img)

    frame_obj = Frame(camera_id="CAM-UNIT", frame_id=1, frame_bytes=buf.tobytes())
    results = detector.detect(frame_obj, image_np=blank_img)
    
    assert isinstance(results, list)
    # Blank image should produce 0 detections safely without crashing
    assert len(results) == 0
