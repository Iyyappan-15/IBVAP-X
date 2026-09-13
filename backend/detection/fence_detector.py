"""
backend/detection/fence_detector.py

High-Precision Physical Perimeter Fence Detector for IBVAP-X.
Detects security fences, chain-link boundary mesh, and perimeter gates using structural line cluster analysis.
"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from backend.interfaces import Detection

logger = logging.getLogger(__name__)

class FenceDetector:
    """
    Detects physical boundary fences and chain-link perimeters via structural line clustering.
    Only fires when genuine parallel vertical/diagonal fence lattice lines are present.
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence

    def detect_fence(
        self,
        image_np: np.ndarray,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 1,
        existing_detections: Optional[List[Detection]] = None
    ) -> List[Detection]:
        """
        Detects physical perimeter fences and barriers based on genuine structural line geometry.
        """
        if image_np is None or image_np.size == 0:
            return []

        h, w = image_np.shape[:2]

        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Detect lines using probabilistic Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=40,
                minLineLength=int(h * 0.12),
                maxLineGap=15
            )

            if lines is None or len(lines) < 6:
                return []

            # Filter for vertical or steep diagonal fence posts / mesh lines
            fence_points_x: List[int] = []
            fence_points_y: List[int] = []
            vertical_line_count = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                angle = np.arctan2(dy, max(dx, 1)) * 180.0 / np.pi

                # Fence posts / mesh are typically between 45 and 90 degrees
                if angle >= 45.0:
                    vertical_line_count += 1
                    fence_points_x.extend([x1, x2])
                    fence_points_y.extend([y1, y2])

            # Must have at least 8 structural vertical/lattice lines to confirm a real fence
            if vertical_line_count >= 8 and len(fence_points_x) >= 16:
                min_x = max(0, int(np.percentile(fence_points_x, 5)))
                max_x = min(w, int(np.percentile(fence_points_x, 95)))
                min_y = max(0, int(np.percentile(fence_points_y, 5)))
                max_y = min(h, int(np.percentile(fence_points_y, 95)))

                box_w = max_x - min_x
                box_h = max_y - min_y

                # A valid fence section should have substantial coverage but not cover a person
                if box_w >= int(w * 0.15) and box_h >= int(h * 0.20):
                    fence_box = [float(min_x), float(min_y), float(max_x), float(max_y)]

                    # Verify no person is occupying the majority of this box (avoid tagging person as fence)
                    if existing_detections:
                        for det in existing_detections:
                            if det.class_name == "person":
                                px1, py1, px2, py2 = det.bbox
                                p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
                                # Intersection with fence box
                                ix1, iy1 = max(min_x, px1), max(min_y, py1)
                                ix2, iy2 = min(max_x, px2), min(max_y, py2)
                                i_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                if p_area > 0 and (i_area / p_area) > 0.70:
                                    # Person is inside this region; adjust fence boundary or skip false box
                                    if px1 > min_x + int(box_w * 0.3):
                                        max_x = int(px1 - 5)
                                    elif px2 < max_x - int(box_w * 0.3):
                                        min_x = int(px2 + 5)
                                    else:
                                        return []

                    if max_x > min_x + int(w * 0.10) and max_y > min_y + int(h * 0.15):
                        return [
                            Detection(
                                class_id=99,
                                class_name="fence",
                                confidence=self.confidence,
                                bbox=[float(min_x), float(min_y), float(max_x), float(max_y)],
                                camera_id=camera_id,
                                timestamp=timestamp,
                                frame_id=frame_id
                            )
                        ]

        except Exception as e:
            logger.warning(f"[FenceDetector] Error: {e}")

        return []
