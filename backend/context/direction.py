from typing import List, Tuple
import numpy as np
from backend.config import settings
from backend.interfaces import DirectionEnum

class DirectionClassifier:
    """Classifies track trajectory movement relative to configured boundary vector."""

    def __init__(self, min_trajectory_frames: int = None):
        self.min_trajectory_frames = min_trajectory_frames or settings.DIRECTION_MIN_FRAMES

    def classify_direction(
        self,
        trajectory: List[Tuple[float, float]],
        boundary_vector: Tuple[float, float] = (0.0, -1.0)
    ) -> DirectionEnum:
        """
        Classifies direction of trajectory relative to boundary vector.
        Returns UNCERTAIN if trajectory is shorter than min_trajectory_frames.
        """
        if len(trajectory) < self.min_trajectory_frames:
            return DirectionEnum.UNCERTAIN

        p_start = np.array(trajectory[0])
        p_end = np.array(trajectory[-1])
        move_vec = p_end - p_start

        norm_move = np.linalg.norm(move_vec)
        norm_bound = np.linalg.norm(boundary_vector)

        if norm_move <= 1e-5 or norm_bound <= 1e-5:
            return DirectionEnum.UNCERTAIN

        # Cosine similarity angle between movement vector and boundary vector
        cos_sim = np.dot(move_vec, boundary_vector) / (norm_move * norm_bound)

        if cos_sim >= 0.5:
            return DirectionEnum.TOWARD_BOUNDARY
        elif cos_sim <= -0.5:
            return DirectionEnum.AWAY
        else:
            return DirectionEnum.LATERAL
