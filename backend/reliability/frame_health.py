from typing import Tuple, Dict, List
import time

class FrameHealthAnalyzer:
    """Tracks frame continuity, drop rates, timestamp gaps, and feed stalls per camera."""

    def __init__(self, expected_fps: float = 30.0, max_gap_seconds: float = 2.0):
        self.expected_fps = expected_fps
        self.max_gap_seconds = max_gap_seconds
        
        # Structure: camera_id -> {"last_timestamp": float, "last_frame_id": int, "gaps": List[float]}
        self.camera_stream_history: Dict[str, Dict] = {}

    def analyze(self, camera_id: str, frame_id: int, timestamp: float) -> Tuple[float, str]:
        """
        Evaluates frame continuity.
        Returns Tuple[score (0-100), reason_message].
        """
        if camera_id not in self.camera_stream_history:
            self.camera_stream_history[camera_id] = {
                "last_timestamp": timestamp,
                "last_frame_id": frame_id,
                "gaps": []
            }
            return 100.0, ""

        hist = self.camera_stream_history[camera_id]
        gap = timestamp - hist["last_timestamp"]
        frame_id_diff = frame_id - hist["last_frame_id"]

        hist["last_timestamp"] = timestamp
        hist["last_frame_id"] = frame_id

        # Track recent gap duration
        hist["gaps"].append(gap)
        if len(hist["gaps"]) > 30:
            hist["gaps"].pop(0)

        reason = ""
        score = 100.0

        if gap > self.max_gap_seconds:
            score = 20.0
            reason = f"Stream stall/freeze detected (Gap: {gap:.2f}s > max {self.max_gap_seconds:.1f}s)"
        elif frame_id_diff > 1:
            dropped = frame_id_diff - 1
            score = max(50.0, 100.0 - (dropped * 10.0))
            reason = f"Frame drop detected ({dropped} frames skipped)"

        return round(score, 1), reason
