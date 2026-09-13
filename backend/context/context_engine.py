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
        timestamp: float = None,
        image_np: Optional[np.ndarray] = None,
        active_tracks: Optional[List[Track]] = None
    ) -> ContextEvent:
        curr_time = timestamp or track.last_seen
        last_center = track.trajectory[-1] if track.trajectory else (0.0, 0.0)

        # Normalize point coordinates to 0.0 - 1.0 fraction
        norm_x = max(0.0, min(1.0, last_center[0] / float(frame_width)))
        norm_y = max(0.0, min(1.0, last_center[1] / float(frame_height)))

        # Check zone entry (Fallback: in simulated feeds or upload, default central area is monitored zone)
        zones = self.camera_zones.get(camera_id, [])
        matched_zone_name: Optional[str] = None
        in_zone = False

        if zones:
            for zm in zones:
                if zm.is_point_in_zone(norm_x, norm_y):
                    in_zone = True
                    matched_zone_name = zm.zone_name
                    break
        else:
            # Default perimeter monitoring zone for upload / unconfigured cameras
            in_zone = True
            matched_zone_name = "Perimeter Buffer Zone Alpha"

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

        # 1. Evaluate Hostile Direct Frontal Approach
        bw = track.bbox[2] - track.bbox[0] if track.bbox else 0.0
        bh = track.bbox[3] - track.bbox[1] if track.bbox else 0.0
        
        # Bounding box expansion detection or close proximity
        is_expanding = False
        if hasattr(track, "initial_bbox") and track.initial_bbox:
            init_h = track.initial_bbox[3] - track.initial_bbox[1]
            if init_h > 0 and (bh / init_h) >= 1.20:
                is_expanding = True

        hostile_approach = bool(
            bh >= float(frame_height) * 0.28 or
            (len(track.trajectory) >= 3 and bh >= float(frame_height) * 0.22) or
            is_expanding
        )

        # Classify direction context (hostile camera approach is directed toward border / post)
        if hostile_approach and d_context in [DirectionEnum.LATERAL, DirectionEnum.UNCERTAIN]:
            d_context = DirectionEnum.TOWARD_BOUNDARY

        # 2. Evaluate Handheld Object / Stone / Weapon Presence
        holding_object = bool(track.class_name in ["stone", "knife", "baseball bat", "weapon", "sports ball"])
        if not holding_object and active_tracks and track.bbox:
            px1, py1, px2, py2 = track.bbox
            for other_t in active_tracks:
                if other_t.track_id != track.track_id and other_t.class_name in ["stone", "knife", "weapon", "sports ball"]:
                    if other_t.trajectory:
                        ox, oy = other_t.trajectory[-1]
                        if (px1 - 60) <= ox <= (px2 + 60) and (py1 - 40) <= oy <= (py2 + 40):
                            holding_object = True
                            break

        # 3. Evaluate Adverse Weather (Fog / Low Visibility Snow / Heavy Mist)
        adverse_weather = False
        if image_np is not None and image_np.size > 0:
            try:
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
                mean_lum = float(np.mean(gray))
                std_lum = float(np.std(gray))
                if (std_lum < 42.0 and mean_lum > 110.0) or (std_lum < 30.0) or (mean_lum < 45.0):
                    adverse_weather = True
            except Exception:
                pass

        # 4. Evaluate Sensor Tampering / Direct Physical Attack on Camera
        tampering_detected = bool(
            (bh >= float(frame_height) * 0.50) or
            (hostile_approach and (holding_object or bh >= float(frame_height) * 0.38))
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
            direction=d_context,
            hostile_approach=hostile_approach,
            holding_object=holding_object,
            adverse_weather=adverse_weather,
            tampering_detected=tampering_detected
        )
