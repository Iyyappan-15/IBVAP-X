from typing import Tuple
import cv2
import numpy as np

class ObstructionAnalyzer:
    """Detects camera lens obstruction or spray-paint using uniform color/intensity patch heuristics."""

    def __init__(self, std_threshold: float = 15.0):
        self.std_threshold = std_threshold

    def analyze(self, image_np: np.ndarray) -> Tuple[float, str]:
        """
        Returns Tuple[score (0-100), reason_message].
        100 = clear view, lower = uniform block/obstruction detected.
        """
        if image_np is None or image_np.size == 0:
            return 0.0, "Missing frame image data"

        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        std_dev = float(np.std(gray))

        reason = ""
        if std_dev < self.std_threshold:
            score = max(0.0, (std_dev / self.std_threshold) * 60.0)
            reason = f"Lens obstruction or covered camera patch detected (Low std dev: {std_dev:.1f} < threshold {self.std_threshold:.1f})"
        else:
            score = 100.0

        return round(score, 1), reason
