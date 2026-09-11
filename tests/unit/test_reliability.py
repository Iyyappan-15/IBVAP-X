import pytest
import numpy as np
import cv2

from backend.reliability.blur import BlurAnalyzer
from backend.reliability.brightness import BrightnessAnalyzer
from backend.reliability.frame_health import FrameHealthAnalyzer
from backend.reliability.obstruction import ObstructionAnalyzer
from backend.reliability.reliability_score import CameraReliabilityEngine, apply_demo_degradation
from backend.interfaces import CameraStatus

def test_blur_analyzer():
    analyzer = BlurAnalyzer(blur_threshold=50.0)
    
    # Sharp image with high texture
    sharp_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    score_sharp, var_sharp, reason_sharp = analyzer.analyze(sharp_img)
    assert score_sharp > 50.0
    assert reason_sharp == ""

    # Heavily blurred image
    blurred_img = cv2.GaussianBlur(sharp_img, (51, 51), 0)
    score_blur, var_blur, reason_blur = analyzer.analyze(blurred_img)
    assert score_blur < score_sharp
    assert "High image blur detected" in reason_blur

def test_brightness_analyzer():
    analyzer = BrightnessAnalyzer(min_brightness=40.0, max_brightness=220.0)

    # Dark image
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8) + 10
    score_dark, mean_dark, reason_dark = analyzer.analyze(dark_img)
    assert score_dark < 50.0
    assert "Low brightness" in reason_dark

    # Optimal image
    normal_img = np.zeros((100, 100, 3), dtype=np.uint8) + 128
    score_norm, mean_norm, reason_norm = analyzer.analyze(normal_img)
    assert score_norm == 100.0
    assert reason_norm == ""

def test_apply_demo_degradation():
    sharp_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    degraded = apply_demo_degradation(sharp_img, blur_ksize=31, brightness_factor=0.2)
    
    assert degraded.shape == sharp_img.shape
    assert np.mean(degraded) < np.mean(sharp_img)

def test_camera_reliability_engine():
    engine = CameraReliabilityEngine()
    sharp_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    result_sharp = engine.calculate_reliability("CAM-01", frame_id=1, timestamp=100.0, image_np=sharp_img)
    assert result_sharp.composite_reliability_score > 70.0
    assert result_sharp.status in [CameraStatus.GOOD, CameraStatus.DEGRADED]

    # Degrade image programmatically
    degraded_img = apply_demo_degradation(sharp_img, blur_ksize=51, brightness_factor=0.2)
    result_degraded = engine.calculate_reliability("CAM-01", frame_id=2, timestamp=100.1, image_np=degraded_img)
    
    assert result_degraded.composite_reliability_score < result_sharp.composite_reliability_score
    assert result_degraded.status in [CameraStatus.POOR, CameraStatus.DEGRADED, CameraStatus.OFFLINE]
    assert len(result_degraded.reasons) > 0
