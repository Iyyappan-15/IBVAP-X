from typing import Tuple
import cv2
import numpy as np
from backend.config import settings

class BlurAnalyzer:
    """Calculates image sharpness using Laplacian variance."""

    def __init__(self, blur_threshold: float = None):
        self.threshold = blur_threshold or settings.BLUR_THRESHOLD

    def analyze(self, image_np: np.ndarray) -> Tuple[float, float, str]:
        """
        Returns Tuple[score (0-100), raw_variance, reason_message].
        100 = perfectly sharp, 0 = severely blurred.
        """
        if image_np is None or image_np.size == 0:
            return 0.0, 0.0, "Camera frame image data missing"

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize score: threshold variance maps to 80 score
        # variance >= 2 * threshold maps to 100 score
        score = min(100.0, max(0.0, (variance / (self.threshold * 2.0)) * 100.0))

        if variance < self.threshold:
            reason = f"High image blur detected (Laplacian variance {variance:.1f} < threshold {self.threshold:.1f})"
        else:
            reason = ""

        return round(score, 1), round(variance, 1), reason
