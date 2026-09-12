"""
backend/detection/fence_detector.py

Fence & Perimeter Barrier Intelligence Module for IBVAP-X.
Analyzes CCTV video frames for physical security fences (chain-link mesh, vertical posts, barbed wire)
using edge-density texture analysis, diamond mesh cross-hatch detection, and Hough line structures.
"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from backend.interfaces import Detection

logger = logging.getLogger(__name__)

class FenceDetector:
    """
    Detects security fences, chain-link barriers, and perimeter posts in surveillance video.
    Returns Detection objects with class_name="fence".
    """

    def __init__(self, min_fence_area_ratio: float = 0.05, confidence: float = 0.85):
        self.min_fence_area_ratio = min_fence_area_ratio
        self.confidence = confidence
        self.cached_fence_bbox: Optional[List[float]] = None
        self.cached_frames = 0

    def detect_fence(
        self,
        image_np: np.ndarray,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 1
    ) -> List[Detection]:
        """
        Analyzes frame texture and line geometry to locate physical perimeter fence structures.
        """
        if image_np is None or image_np.size == 0:
            return []

        h, w = image_np.shape[:2]
        total_area = h * w

        # Every 10 frames, recompute or use temporal smoothing
        if self.cached_fence_bbox is not None and self.cached_frames < 30:
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
            # Focus on mid-to-right or perimeter regions where fences typically stand
            # Apply adaptive threshold / Canny edge
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Detect vertical and diagonal mesh patterns
            # Morphological kernels for vertical posts and cross-hatch mesh
            kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 15))
            vert_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_vert)

            kernel_mesh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mesh_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_mesh)

            combined = cv2.addWeighted(vert_edges, 0.5, mesh_edges, 0.5, 0)

            # Find bounding contours of dense edge regions
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            fence_boxes = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh

                # Check if region has significant height (like a fence) and edge density
                if area >= total_area * self.min_fence_area_ratio and bh >= h * 0.40:
                    roi = edges[y:y+bh, x:x+bw]
                    edge_density = np.count_nonzero(roi) / float(area)

                    # Chain link fence has high edge density (> 0.04)
                    if edge_density > 0.035:
                        fence_boxes.append([float(x), float(y), float(x + bw), float(y + bh)])

            if fence_boxes:
                # Merge overlapping fence boxes
                min_x = min(b[0] for b in fence_boxes)
                min_y = min(b[1] for b in fence_boxes)
                max_x = max(b[2] for b in fence_boxes)
                max_y = max(b[3] for b in fence_boxes)

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
            logger.warning(f"[FenceDetector] Error during fence texture analysis: {e}")

        return []
