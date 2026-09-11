import math
from typing import List, Tuple
from shapely.geometry import Polygon

class LocalFOVCalculator:
    """Computes field-of-view (FOV) wedge polygons in local Euclidean metric space (x, y meters)."""

    @staticmethod
    def calculate_fov_wedge_metric(
        origin_xy: Tuple[float, float],
        heading_degrees: float,
        fov_degrees: float,
        range_meters: float,
        num_arc_pts: int = 15
    ) -> Polygon:
        """
        Calculates FOV wedge polygon in local Euclidean metric plane (meters).
        origin_xy: (x_meters, y_meters)
        heading_degrees: 0 = North (+Y), 90 = East (+X)
        """
        ox, oy = origin_xy
        start_angle = heading_degrees - (fov_degrees / 2.0)
        end_angle = heading_degrees + (fov_degrees / 2.0)

        # Polygon vertices starting at camera origin
        vertices = [(ox, oy)]

        # Generate arc points along range radius
        angles = [start_angle + (i / float(num_arc_pts - 1)) * (end_angle - start_angle) for i in range(num_arc_pts)]
        for angle_deg in angles:
            rad = math.radians(angle_deg)
            # Heading 0 deg = North (+Y), 90 deg = East (+X)
            px = ox + range_meters * math.sin(rad)
            py = oy + range_meters * math.cos(rad)
            vertices.append((px, py))

        # Close polygon back to origin
        vertices.append((ox, oy))
        return Polygon(vertices)
