import json
import os
import logging
from typing import List, Dict, Optional
from backend.interfaces import Track, ContextEvent, TimeContextEnum, DirectionEnum
from backend.context.zones import ZoneManager
from backend.context.loitering import LoiteringDetector
from backend.context.time_context import TimeContextClassifier
from backend.context.direction import DirectionClassifier

logger = logging.getLogger(__name__)

class ContextEngine:
    """Aggregates spatial, loitering, temporal, and directional analysis into ContextEvents."""

    def __init__(self, config_path: str = "data/config/cameras.json"):
        self.config_path = config_path
        self.camera_zones: Dict[str, List[ZoneManager]] = {}
        self.camera_boundaries: Dict[str, tuple] = {}
        self.loitering_detectors: Dict[str, LoiteringDetector] = {}
        self.time_classifier = TimeContextClassifier()
        self.direction_classifier = DirectionClassifier()

        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"[ContextEngine] Config file '{self.config_path}' not found. Using empty zones.")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cam in data.get("cameras", []):
            cam_id = cam["camera_id"]
            self.camera_boundaries[cam_id] = tuple(cam.get("boundary_vector", [0.0, -1.0]))
            self.camera_zones[cam_id] = []
            self.loitering_detectors[cam_id] = LoiteringDetector()

            for z in cam.get("zones", []):
                zm = ZoneManager(
                    zone_name=z["zone_name"],
                    polygon_normalized=z["polygon_normalized"]
                )
                self.camera_zones[cam_id].append(zm)

    def analyze_track(
        self,
        camera_id: str,
        track: Track,
        frame_width: int = 640,
        frame_height: int = 480,
        timestamp: float = None
    ) -> ContextEvent:
        curr_time = timestamp or track.last_seen
        last_center = track.trajectory[-1] if track.trajectory else (0.0, 0.0)

        # Normalize point coordinates to 0.0 - 1.0 fraction
        norm_x = max(0.0, min(1.0, last_center[0] / float(frame_width)))
        norm_y = max(0.0, min(1.0, last_center[1] / float(frame_height)))

        # Check zone entry
        zones = self.camera_zones.get(camera_id, [])
        matched_zone_name: Optional[str] = None
        in_zone = False

        for zm in zones:
            if zm.is_point_in_zone(norm_x, norm_y):
                in_zone = True
                matched_zone_name = zm.zone_name
                break

        # Check loitering
        loitering_detector = self.loitering_detectors.setdefault(camera_id, LoiteringDetector())
        is_loitering, duration = loitering_detector.update_track_zone(
            track_id=track.track_id,
            zone_name=matched_zone_name,
            current_time=curr_time
        )

        # Classify time context
        t_context = self.time_classifier.classify(curr_time)

        # Classify direction context
        b_vec = self.camera_boundaries.get(camera_id, (0.0, -1.0))
        d_context = self.direction_classifier.classify_direction(
            trajectory=track.trajectory,
            boundary_vector=b_vec
        )

        return ContextEvent(
            camera_id=camera_id,
            track_id=track.track_id,
            class_name=track.class_name,
            timestamp=curr_time,
            in_restricted_zone=in_zone,
            zone_name=matched_zone_name,
            loitering=is_loitering,
            loitering_duration_seconds=round(duration, 2),
            time_context=t_context,
            direction=d_context
        )
