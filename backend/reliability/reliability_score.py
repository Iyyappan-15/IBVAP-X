import time
import logging
from typing import List, Optional
import cv2
import numpy as np

from backend.config import settings
from backend.interfaces import ReliabilityScore, CameraStatus, Frame
from backend.reliability.blur import BlurAnalyzer
from backend.reliability.brightness import BrightnessAnalyzer
from backend.reliability.frame_health import FrameHealthAnalyzer
from backend.reliability.obstruction import ObstructionAnalyzer

logger = logging.getLogger(__name__)

def apply_demo_degradation(image_np: np.ndarray, blur_ksize: int = 51, brightness_factor: float = 0.3) -> np.ndarray:
    """Programmatically degrades a frame image with Gaussian blur and darkness attenuation for demo presentation."""
    if image_np is None or image_np.size == 0:
        return image_np

    degraded = image_np.copy()
    if blur_ksize > 1:
        # Ensure ksize is odd
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        degraded = cv2.GaussianBlur(degraded, (blur_ksize, blur_ksize), 0)

    if brightness_factor < 1.0:
        degraded = np.clip(degraded.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    return degraded


class CameraReliabilityEngine:
    """Computes composite camera reliability score from independent sub-metric analyzers."""

    def __init__(self):
        self.blur_analyzer = BlurAnalyzer()
        self.brightness_analyzer = BrightnessAnalyzer()
        self.frame_analyzer = FrameHealthAnalyzer()
        self.obstruction_analyzer = ObstructionAnalyzer()

        # Configurable weights from settings (.env)
        self.w_blur = settings.RELIABILITY_WEIGHT_BLUR
        self.w_brightness = settings.RELIABILITY_WEIGHT_BRIGHTNESS
        self.w_frame = settings.RELIABILITY_WEIGHT_FRAME
        self.w_obstruction = settings.RELIABILITY_WEIGHT_OBSTRUCTION

    def calculate_reliability(
        self,
        camera_id: str,
        frame_id: int,
        timestamp: float,
        image_np: Optional[np.ndarray] = None
    ) -> ReliabilityScore:
        reasons: List[str] = []

        if image_np is None or image_np.size == 0:
            return ReliabilityScore(
                camera_id=camera_id,
                timestamp=timestamp or time.time(),
                blur_score=0.0,
                brightness_score=0.0,
                frame_health_score=0.0,
                obstruction_score=0.0,
                composite_reliability_score=0.0,
                status=CameraStatus.OFFLINE,
                reasons=["No video frame data available (Camera offline)"]
            )

        # 1. Blur Analysis
        b_score, b_raw, b_reason = self.blur_analyzer.analyze(image_np)
        if b_reason:
            reasons.append(b_reason)

        # 2. Brightness Analysis
        br_score, br_raw, br_reason = self.brightness_analyzer.analyze(image_np)
        if br_reason:
            reasons.append(br_reason)

        # 3. Frame Health Analysis
        fh_score, fh_reason = self.frame_analyzer.analyze(camera_id, frame_id, timestamp)
        if fh_reason:
            reasons.append(fh_reason)

        # 4. Obstruction Analysis
        obs_score, obs_reason = self.obstruction_analyzer.analyze(image_np)
        if obs_reason:
            reasons.append(obs_reason)

        # Calculate composite weighted reliability score
        composite = (
            (b_score * self.w_blur) +
            (br_score * self.w_brightness) +
            (fh_score * self.w_frame) +
            (obs_score * self.w_obstruction)
        )
        composite = round(min(100.0, max(0.0, composite)), 1)

        # Determine Camera Status classification
        if composite >= settings.RELIABILITY_GOOD_THRESHOLD:
            status = CameraStatus.GOOD
        elif composite >= settings.RELIABILITY_DEGRADED_THRESHOLD:
            status = CameraStatus.DEGRADED
        elif composite >= settings.RELIABILITY_POOR_THRESHOLD:
            status = CameraStatus.POOR
        else:
            status = CameraStatus.OFFLINE

        return ReliabilityScore(
            camera_id=camera_id,
            timestamp=timestamp,
            blur_score=b_score,
            brightness_score=br_score,
            frame_health_score=fh_score,
            obstruction_score=obs_score,
            composite_reliability_score=composite,
            status=status,
            reasons=reasons
        )
