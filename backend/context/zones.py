from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon
import cv2
import numpy as np

class ZoneManager:
    """Manages geometric polygon zones using normalized 0.0-1.0 coordinates."""

    def __init__(self, zone_name: str, polygon_normalized: List[List[float]]):
        self.zone_name = zone_name
        self.polygon_normalized = polygon_normalized
        self.shapely_poly = Polygon(polygon_normalized)

    def is_point_in_zone(self, norm_x: float, norm_y: float) -> bool:
        """Checks if normalized point (0.0 to 1.0) is inside the polygon zone."""
        point = Point(norm_x, norm_y)
        return self.shapely_poly.contains(point) or self.shapely_poly.touches(point)

    def draw_zone_overlay(self, image_np: np.ndarray, color=(0, 0, 255), thickness=2) -> np.ndarray:
        """Draws zone polygon over image canvas."""
        h, w = image_np.shape[:2]
        pts = np.array(
            [[int(pt[0] * w), int(pt[1] * h)] for pt in self.polygon_normalized],
            dtype=np.int32
        ).reshape((-1, 1, 2))

        annotated = image_np.copy()
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Semi-transparent fill
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

        return annotated
