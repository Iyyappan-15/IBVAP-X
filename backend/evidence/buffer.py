from typing import List, collections
import cv2
import numpy as np
from backend.interfaces import Frame

class PreEventRingBuffer:
    """Bounded circular ring buffer storing recent frames in RAM for pre-event evidence capture."""

    def __init__(self, max_seconds: int = 10, fps: float = 10.0):
        self.max_frames = int(max_seconds * fps)
        self.buffer = collections.deque(maxlen=self.max_frames)

    def add_frame(self, frame_obj: Frame, image_np: np.ndarray):
        """Adds a frame object and image numpy array to circular buffer."""
        self.buffer.append((frame_obj, image_np.copy()))

    def get_buffered_frames(self) -> List[tuple]:
        """Returns all buffered pre-event (frame_obj, image_np) tuples in chronological order."""
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
