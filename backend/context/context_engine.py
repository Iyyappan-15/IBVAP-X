import json
import os
import logging
from typing import List, Dict, Optional
import numpy as np
import cv2
from backend.interfaces import Track, ContextEvent, TimeContextEnum, DirectionEnum, ReliabilityScore
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
        active_tracks: Optional[List[Track]] = None,
        reliability_score: Optional[ReliabilityScore] = None
    ) -> ContextEvent:
        curr_time = timestamp or track.last_seen
        last_center = track.trajectory[-1] if track.trajectory else (0.0, 0.0)

        # Normalize point coordinates to 0.0 - 1.0 fraction
        norm_x = max(0.0, min(1.0, last_center[0] / float(frame_width)))
        norm_y = max(0.0, min(1.0, last_center[1] / float(frame_height)))

        # Check zone entry (Configured restricted polygons only)
        zones = self.camera_zones.get(camera_id, [])
        matched_zone_name: Optional[str] = None
        in_zone = False

        if zones:
            for zm in zones:
                if zm.is_point_in_zone(norm_x, norm_y):
                    matched_zone_name = zm.zone_name
                    z_lower = zm.zone_name.lower()
                    if "primary" in z_lower or "observation" in z_lower or "general" in z_lower or "safe" in z_lower:
                        in_zone = False
                    else:
                        in_zone = True
                    break
        else:
            # Uploaded or unconfigured feeds without explicit restricted polygons remain outside restricted zones
            in_zone = False
            matched_zone_name = "General Observation Sector"

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
        
        # Requires kinematic trajectory: target must be approaching camera plane (expanding + moving)
        hostile_approach = False
        is_approach_target = track.class_name in ["person", "car", "truck", "motorcycle", "vehicle"]
        
        if is_approach_target and len(track.trajectory) >= 4 and hasattr(track, "initial_bbox") and track.initial_bbox:
            init_h = track.initial_bbox[3] - track.initial_bbox[1]
            first_pt = track.trajectory[0]
            last_pt = track.trajectory[-1]
            disp_y = last_pt[1] - first_pt[1]
            total_disp = float(np.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1]))
            
            # Hostile approach: rapid bounding box expansion towards camera post AND non-trivial downward displacement
            if init_h > 0:
                expansion_ratio = bh / init_h
                if expansion_ratio >= 1.35 and disp_y >= 20.0 and total_disp >= 25.0 and bh >= float(frame_height) * 0.35:
                    hostile_approach = True

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
        # Physical sensor tampering requires actual sensor degradation (blur/obstruction) while in close proximity
        tampering_detected = False
        if reliability_score is not None:
            blur_score = getattr(reliability_score, "blur_score", 100.0)
            obs_score = getattr(reliability_score, "obstruction_score", 100.0)
            comp_rel = getattr(reliability_score, "composite_reliability_score", 100.0)

            # Sensor degradation must be genuinely observed on the camera feed
            if obs_score < 40.0 or blur_score < 25.0 or comp_rel < 50.0:
                if bh >= float(frame_height) * 0.50 or holding_object or hostile_approach:
                    tampering_detected = True
        else:
            # Fallback if reliability score omitted: only extreme close-contact with confirmed weapon/projectile
            if hostile_approach and holding_object and bh >= float(frame_height) * 0.55:
                tampering_detected = True

        # 5. Camera Broken Detection — Physical destruction of lens/housing
        # Indicators:
        #   a) Severe blur (Laplacian variance near 0, blur_score < 15)  → lens cracked/hit
        #   b) Near-complete obstruction (obstruction_score < 20)         → lens fully covered
        #   c) Hostile approach with object AND camera already degrading   → escalate to broken
        #   d) Sudden full-frame darkness (mean_lum < 20, std < 12)       → camera dead
        camera_broken = False
        if reliability_score is not None:
            blur_score = reliability_score.blur_score
            obs_score = reliability_score.obstruction_score
            composite = reliability_score.composite_reliability_score

            # Lens cracked or heavily damaged → blur near zero
            if blur_score < 15.0:
                camera_broken = True
            # Lens fully covered / spray-painted / smashed → near uniform patch
            elif obs_score < 20.0:
                camera_broken = True
            # Composite below 25 while hostile attack is ongoing → camera is being broken
            elif composite < 25.0 and (hostile_approach or tampering_detected):
                camera_broken = True

        # Also check raw image for sudden pitch-black or all-white static frame
        if not camera_broken and image_np is not None and image_np.size > 0:
            try:
                gray_check = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
                mean_raw = float(np.mean(gray_check))
                std_raw = float(np.std(gray_check))
                # Pitch dark (camera dead) or completely white (shattered lens glare)
                if (mean_raw < 15.0 and std_raw < 10.0) or (mean_raw > 240.0 and std_raw < 10.0):
                    camera_broken = True
                # Static noise / heavy flicker: very high uniform variance
                if std_raw > 90.0 and mean_raw < 100.0:
                    camera_broken = True
            except Exception:
                pass

        # If camera is confirmed broken, always set tampering_detected as well
        if camera_broken:
            tampering_detected = True

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
            tampering_detected=tampering_detected,
            camera_broken=camera_broken
        )
