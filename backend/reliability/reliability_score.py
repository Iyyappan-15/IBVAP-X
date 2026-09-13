import time
import logging
from typing import List, Optional, Dict
import cv2
import numpy as np

from backend.config import settings
from backend.interfaces import ReliabilityScore, CameraStatus, Frame
from backend.reliability.blur import BlurAnalyzer
from backend.reliability.brightness import BrightnessAnalyzer
from backend.reliability.frame_health import FrameHealthAnalyzer
from backend.reliability.obstruction import ObstructionAnalyzer

logger = logging.getLogger(__name__)

def apply_demo_degradation(image_np: np.ndarray, blur_ksize: int = 51, brightness_factor: float = 0.3) -> np.ndarray:
    """Programmatically degrades a frame image with Gaussian blur and darkness attenuation for demo presentation."""
    if image_np is None or image_np.size == 0:
        return image_np

    degraded = image_np.copy()
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        degraded = cv2.GaussianBlur(degraded, (blur_ksize, blur_ksize), 0)

    if brightness_factor < 1.0:
        degraded = np.clip(degraded.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    return degraded


class CameraReliabilityEngine:
    """
    Computes composite camera reliability score from independent sub-metric analyzers
    using a rolling window and adaptive temporal hysteresis to ensure smooth, stable status transitions.
    """

    def __init__(self):
        self.blur_analyzer = BlurAnalyzer()
        self.brightness_analyzer = BrightnessAnalyzer()
        self.frame_analyzer = FrameHealthAnalyzer()
        self.obstruction_analyzer = ObstructionAnalyzer()

        # Configurable weights from settings (.env)
        self.w_blur = settings.RELIABILITY_WEIGHT_BLUR
        self.w_brightness = settings.RELIABILITY_WEIGHT_BRIGHTNESS
        self.w_frame = settings.RELIABILITY_WEIGHT_FRAME
        self.w_obstruction = settings.RELIABILITY_WEIGHT_OBSTRUCTION

        self.window_size = settings.RELIABILITY_WINDOW_FRAMES
        self.hysteresis_frames = settings.RELIABILITY_HYSTERESIS_FRAMES

        # Rolling history state per camera
        # camera_id -> {"blur": [], "brightness": [], "frame": [], "obstruction": [], "composite": []}
        self.history: Dict[str, Dict[str, List[float]]] = {}
        # camera_id -> {"confirmed_status": CameraStatus, "candidate_status": CameraStatus, "candidate_count": int}
        self.status_state: Dict[str, Dict] = {}

    def calculate_reliability(
        self,
        camera_id: str,
        frame_id: int,
        timestamp: float,
        image_np: Optional[np.ndarray] = None
    ) -> ReliabilityScore:
        reasons: List[str] = []

        if image_np is None or image_np.size == 0:
            raw_b, raw_br, raw_fh, raw_obs, raw_comp = 0.0, 0.0, 0.0, 0.0, 0.0
            reasons.append("No video frame data available (Camera offline)")
        else:
            # 1. Blur Analysis
            b_score, b_raw, b_reason = self.blur_analyzer.analyze(image_np)
            if b_reason:
                reasons.append(b_reason)

            # 2. Brightness Analysis
            br_score, br_raw, br_reason = self.brightness_analyzer.analyze(image_np)
            if br_reason:
                reasons.append(br_reason)

            # 3. Frame Health Analysis
            fh_score, fh_reason = self.frame_analyzer.analyze(camera_id, frame_id, timestamp)
            if fh_reason:
                reasons.append(fh_reason)

            # 4. Obstruction Analysis
            obs_score, obs_reason = self.obstruction_analyzer.analyze(image_np)
            if obs_reason:
                reasons.append(obs_reason)

            # Calculate raw composite weighted reliability score
            raw_comp = (
                (b_score * self.w_blur) +
                (br_score * self.w_brightness) +
                (fh_score * self.w_frame) +
                (obs_score * self.w_obstruction)
            )
            raw_b, raw_br, raw_fh, raw_obs = b_score, br_score, fh_score, obs_score

        # Determine raw status directly for this frame
        if raw_comp >= settings.RELIABILITY_GOOD_THRESHOLD:
            raw_status = CameraStatus.GOOD
        elif raw_comp >= settings.RELIABILITY_DEGRADED_THRESHOLD:
            raw_status = CameraStatus.DEGRADED
        elif raw_comp >= settings.RELIABILITY_POOR_THRESHOLD:
            raw_status = CameraStatus.POOR
        else:
            raw_status = CameraStatus.OFFLINE

        # Initialize rolling history for camera if absent
        if camera_id not in self.history:
            self.history[camera_id] = {
                "blur": [], "brightness": [], "frame": [], "obstruction": [], "composite": []
            }
            self.status_state[camera_id] = {
                "confirmed_status": raw_status,
                "candidate_status": raw_status,
                "candidate_count": 0
            }

        cam_hist = self.history[camera_id]
        cam_hist["blur"].append(raw_b)
        cam_hist["brightness"].append(raw_br)
        cam_hist["frame"].append(raw_fh)
        cam_hist["obstruction"].append(raw_obs)
        cam_hist["composite"].append(raw_comp)

        # Trim to window size
        for k in cam_hist:
            if len(cam_hist[k]) > self.window_size:
                cam_hist[k].pop(0)

        # Compute rolling window averages
        roll_b = round(float(np.mean(cam_hist["blur"])), 1)
        roll_br = round(float(np.mean(cam_hist["brightness"])), 1)
        roll_fh = round(float(np.mean(cam_hist["frame"])), 1)
        roll_obs = round(float(np.mean(cam_hist["obstruction"])), 1)
        roll_composite = round(float(np.mean(cam_hist["composite"])), 1)

        # Determine candidate status from rolling composite
        if roll_composite >= settings.RELIABILITY_GOOD_THRESHOLD:
            cand_status = CameraStatus.GOOD
        elif roll_composite >= settings.RELIABILITY_DEGRADED_THRESHOLD:
            cand_status = CameraStatus.DEGRADED
        elif roll_composite >= settings.RELIABILITY_POOR_THRESHOLD:
            cand_status = CameraStatus.POOR
        else:
            cand_status = CameraStatus.OFFLINE

        st_data = self.status_state[camera_id]

        # For initial frames (history length <= 3), accept status immediately without hysteresis delay
        if len(cam_hist["composite"]) <= 3:
            st_data["confirmed_status"] = cand_status
            st_data["candidate_status"] = cand_status
            st_data["candidate_count"] = 0
        else:
            if cand_status == st_data["confirmed_status"]:
                st_data["candidate_status"] = cand_status
                st_data["candidate_count"] = 0
            else:
                if cand_status == st_data["candidate_status"]:
                    st_data["candidate_count"] += 1
                else:
                    st_data["candidate_status"] = cand_status
                    st_data["candidate_count"] = 1

                # Transition confirmed if candidate count reaches threshold (or immediately if OFFLINE)
                required = 2 if cand_status in [CameraStatus.OFFLINE, CameraStatus.POOR] else self.hysteresis_frames
                if st_data["candidate_count"] >= required:
                    st_data["confirmed_status"] = cand_status
                    st_data["candidate_count"] = 0

        final_status = st_data["confirmed_status"]

        return ReliabilityScore(
            camera_id=camera_id,
            timestamp=timestamp,
            blur_score=roll_b,
            brightness_score=roll_br,
            frame_health_score=roll_fh,
            obstruction_score=roll_obs,
            composite_reliability_score=roll_composite,
            status=final_status,
            reasons=reasons
        )
