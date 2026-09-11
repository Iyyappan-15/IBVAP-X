from typing import Tuple
import cv2
import numpy as np
from backend.config import settings

class BrightnessAnalyzer:
    """Calculates image brightness statistics from grayscale mean intensity."""

    def __init__(self, min_brightness: float = None, max_brightness: float = None):
        self.min_brightness = min_brightness or settings.BRIGHTNESS_MIN
        self.max_brightness = max_brightness or settings.BRIGHTNESS_MAX

    def analyze(self, image_np: np.ndarray) -> Tuple[float, float, str]:
        """
        Returns Tuple[score (0-100), mean_brightness (0-255), reason_message].
        100 = optimal lighting, lower = too dark or glare overexposure.
        """
        if image_np is None or image_np.size == 0:
            return 0.0, 0.0, "Frame image missing for brightness check"

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        mean_val = float(np.mean(gray))

        reason = ""
        if mean_val < self.min_brightness:
            score = max(0.0, (mean_val / self.min_brightness) * 70.0)
            reason = f"Low brightness/underexposure detected (Mean: {mean_val:.1f} < min {self.min_brightness:.1f})"
        elif mean_val > self.max_brightness:
            score = max(0.0, 100.0 - ((mean_val - self.max_brightness) / (255.0 - self.max_brightness)) * 70.0)
            reason = f"High brightness/glare overexposure detected (Mean: {mean_val:.1f} > max {self.max_brightness:.1f})"
        else:
            score = 100.0

        return round(score, 1), round(mean_val, 1), reason
