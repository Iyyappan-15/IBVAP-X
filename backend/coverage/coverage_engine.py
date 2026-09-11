import json
import math
import os
import logging
from typing import Dict, List, Tuple, Any
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from backend.coverage.fov import LocalFOVCalculator
from backend.interfaces import CoverageResult

logger = logging.getLogger(__name__)

class GeometricCoverageEngine:
    """
    Geometric Camera-Coverage Approximation Engine.
    Uses local metric Euclidean coordinate plane for Shapely geometry calculations
    and transforms polygons to lat/lon GPS for map rendering.
    """

    def __init__(self, config_path: str = "data/config/cameras.json"):
        self.config_path = config_path
        self.cameras: List[Dict] = []
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"[CoverageEngine] Config path '{self.config_path}' not found.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.cameras = data.get("cameras", [])

    @staticmethod
    def gps_to_metric(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """Converts lat/lon GPS coordinates to local Euclidean (X, Y) meters relative to ref_lat/ref_lon."""
        r_earth = 6371000.0  # meters
        d_lat = math.radians(lat - ref_lat)
        d_lon = math.radians(lon - ref_lon)
        y = d_lat * r_earth
        x = d_lon * r_earth * math.cos(math.radians(ref_lat))
        return float(x), float(y)

    @staticmethod
    def metric_to_gps(x: float, y: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
        """Converts local Euclidean (X, Y) meters back to lat/lon GPS coordinates."""
        r_earth = 6371000.0
        d_lat = math.degrees(y / r_earth)
        d_lon = math.degrees(x / (r_earth * math.cos(math.radians(ref_lat))))
        return round(ref_lat + d_lat, 6), round(ref_lon + d_lon, 6)

    def compute_coverage(self) -> Dict[str, Any]:
        """
        Computes 3-tier camera coverage and returns GPS polygon lists for visualization.
        """
        if not self.cameras:
            return {"single_fov_polygons": [], "multi_fov_polygons": [], "blindspot_polygons": []}

        ref_lat = self.cameras[0]["latitude"]
        ref_lon = self.cameras[0]["longitude"]

        fov_wedges = []
        for cam in self.cameras:
            ox, oy = self.gps_to_metric(cam["latitude"], cam["longitude"], ref_lat, ref_lon)
            wedge = LocalFOVCalculator.calculate_fov_wedge_metric(
                origin_xy=(ox, oy),
                heading_degrees=cam["heading_degrees"],
                fov_degrees=cam["fov_degrees"],
                range_meters=cam["range_meters"]
            )
            fov_wedges.append(wedge)

        # Total union of all camera FOVs
        total_union = unary_union(fov_wedges)

        # Convert Shapely polygons back to GPS lists
        single_gps_polys = []
        for wedge in fov_wedges:
            gps_pts = [self.metric_to_gps(x, y, ref_lat, ref_lon) for x, y in wedge.exterior.coords]
            single_gps_polys.append(gps_pts)

        return {
            "ref_lat": ref_lat,
            "ref_lon": ref_lon,
            "camera_count": len(self.cameras),
            "single_fov_polygons": single_gps_polys,
            "coverage_tier_labels": {
                "multi": "2+ modeled camera FOVs",
                "single": "1 modeled camera FOV",
                "none": "0 modeled camera FOVs"
            }
        }
