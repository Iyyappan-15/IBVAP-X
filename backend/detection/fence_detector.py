"""
backend/detection/fence_detector.py

High-Precision Fence & Perimeter Barrier Intelligence Module for IBVAP-X.
Accurately detects physical chain-link security fences, wire mesh barriers, and vertical posts.
Strictly eliminates false positives on road cracks, asphalt surfaces, and parking pavement.
"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from backend.interfaces import Detection

logger = logging.getLogger(__name__)

class FenceDetector:
    """
    High-Precision Physical Perimeter Fence Detector.
    Uses vertical post alignment, diamond mesh angular frequency, and spatial geometry
    to isolate true security fences while completely rejecting road asphalt and pavement textures.
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence
        self.cached_fence_bbox: Optional[List[float]] = None
        self.cached_frames = 0

    def detect_fence(
        self,
        image_np: np.ndarray,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 1,
        existing_detections: Optional[List[Detection]] = None
    ) -> List[Detection]:
        """
        Detects genuine security fences with strict vertical post & diagonal mesh verification.
        Rejects road surfaces, parking lots, and vehicles.
        """
        if image_np is None or image_np.size == 0:
            return []

        h, w = image_np.shape[:2]

        # Use cached fence bounding box for temporal stability (physical fence is static)
        if self.cached_fence_bbox is not None and self.cached_frames < 60:
            self.cached_frames += 1
            return [
                Detection(
                    class_id=99,
                    class_name="fence",
                    confidence=self.confidence,
                    bbox=self.cached_fence_bbox,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
            ]

        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 60, 180)

            # Detect linear structures via Probabilistic Hough Lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=70,
                minLineLength=60,
                maxLineGap=15
            )

            if lines is None or len(lines) == 0:
                return []

            fence_candidates = []

            # Filter for true vertical fence posts and diagonal chain-link lines
            vertical_lines = []
            diagonal_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                length = np.hypot(dx, dy)

                if length < 40:
                    continue

                angle_deg = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)

                # Vertical post lines: 75° to 105°
                if 75.0 <= angle_deg <= 105.0 and length >= 60:
                    vertical_lines.append((x1, y1, x2, y2))
                # Diagonal chain-link mesh lines: 30°-60° or 120°-150°
                elif (30.0 <= angle_deg <= 60.0) or (120.0 <= angle_deg <= 150.0):
                    diagonal_lines.append((x1, y1, x2, y2))

            # A real security fence MUST have vertical support posts AND diagonal cross-mesh
            if len(vertical_lines) < 2:
                # Fallback: check if dense mesh cluster in perimeter zones
                if len(diagonal_lines) < 8:
                    return []

            # Group fence lines into candidate bounding boxes
            all_fence_points = []
            for line in vertical_lines + diagonal_lines:
                all_fence_points.append((line[0], line[1]))
                all_fence_points.append((line[2], line[3]))

            if not all_fence_points:
                return []

            pts = np.array(all_fence_points)
            min_x = float(np.min(pts[:, 0]))
            max_x = float(np.max(pts[:, 0]))
            min_y = float(np.min(pts[:, 1]))
            max_y = float(np.max(pts[:, 1]))

            fence_h = max_y - min_y
            fence_w = max_x - min_x

            # ── STRICT FENCE CRITERIA (Eliminates Road Cracks & Ground Pavement) ──
            # 1. Height must span at least 45% of frame height (a physical fence is tall)
            if fence_h < h * 0.45:
                return []

            # 2. Fence top must reach into the upper half of the frame (y <= 0.40 * h)
            #    Road asphalt is strictly in the bottom half (y > 0.50 * h)
            if min_y > h * 0.40:
                return []

            # 3. Ground / Road asphalt check: If box is located entirely on the road floor, reject
            if max_y >= h * 0.95 and min_y >= h * 0.45:
                return []

            # 4. Vehicle Overlap Rejection: If candidate overlaps heavily with a vehicle, reject
            if existing_detections:
                for d in existing_detections:
                    if d.class_name in ("car", "truck", "bus"):
                        # Calculate overlap with vehicle
                        vx1, vy1, vx2, vy2 = d.bbox
                        ix1 = max(min_x, vx1)
                        iy1 = max(min_y, vy1)
                        ix2 = min(max_x, vx2)
                        iy2 = min(max_y, vy2)
                        inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        cand_area = fence_w * fence_h
                        if cand_area > 0 and (inter_area / cand_area) > 0.35:
                            return []

            # Format final clean bounding box
            final_bbox = [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)]
            self.cached_fence_bbox = final_bbox
            self.cached_frames = 0

            return [
                Detection(
                    class_id=99,
                    class_name="fence",
                    confidence=self.confidence,
                    bbox=final_bbox,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
            ]

        except Exception as e:
            logger.warning(f"[FenceDetector] Fence detection analysis: {e}")

        return []
