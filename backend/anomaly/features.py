from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from backend.config import settings
from backend.interfaces import Track

class TrajectoryFeatureExtractor:
    """Extracts trajectory motion features from Track objects for unsupervised anomaly detection."""

    def __init__(self, min_frames: int = None):
        self.min_frames = min_frames or settings.TRACK_MIN_TRAJECTORY_FRAMES

    def extract_features(self, track: Track) -> Optional[np.ndarray]:
        """
        Extracts 8-dimensional motion feature vector from track trajectory.
        Returns None if trajectory length < min_frames.
        """
        trajectory = track.trajectory
        if not trajectory or len(trajectory) < self.min_frames:
            return None

        coords = np.array(trajectory, dtype=np.float32)
        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        total_distance = float(np.sum(distances))
        avg_speed = float(np.mean(distances)) if len(distances) > 0 else 0.0
        max_speed = float(np.max(distances)) if len(distances) > 0 else 0.0

        # Fraction of frames with near-zero movement (stop ratio)
        stop_count = int(np.sum(distances < 1.0))
        stop_ratio = float(stop_count / float(len(distances))) if len(distances) > 0 else 0.0

        # Direction changes (angles between consecutive vectors)
        direction_changes = 0
        if len(diffs) >= 2:
            for i in range(len(diffs) - 1):
                v1, v2 = diffs[i], diffs[i+1]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-3 and n2 > 1e-3:
                    cos_a = np.dot(v1, v2) / (n1 * n2)
                    if cos_a < 0.5:  # Angle > 60 degrees
                        direction_changes += 1

        # Trajectory curvature (direct distance / path length)
        direct_dist = float(np.linalg.norm(coords[-1] - coords[0]))
        curvature = float(direct_dist / total_distance) if total_distance > 1e-3 else 1.0

        zone_transitions = len(track.zone_history)
        duration_sec = float(track.last_seen - track.start_time)

        # Feature vector shape (8,)
        feature_vector = np.array([
            avg_speed,
            max_speed,
            stop_ratio,
            total_distance,
            float(direction_changes),
            curvature,
            float(zone_transitions),
            duration_sec
        ], dtype=np.float32)

        return feature_vector
