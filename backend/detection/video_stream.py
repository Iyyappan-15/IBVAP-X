from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
import time
import logging
import cv2
import numpy as np

from backend.interfaces import Frame
from backend.detection.schemas import VideoFrameData

logger = logging.getLogger(__name__)

class VideoSourceError(Exception):
    """Custom exception raised when a video stream encounters an unrecoverable error."""
    pass

class VideoSource(ABC):
    """Abstract Base Class for Video Sources."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.frame_count = 0
        self.is_connected = False

    @abstractmethod
    def get_frame(self) -> Optional[Frame]:
        """Retrieve the next frame from the source."""
        pass

    @abstractmethod
    def release(self):
        """Release underlying hardware or file resources."""
        pass


class FileVideoSource(VideoSource):
    """Video Source implementation for Uploaded MP4/AVI Files."""

    def __init__(self, file_path: str, camera_id: str = "CAM-FILE"):
        super().__init__(camera_id=camera_id)
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise VideoSourceError(f"Video file not found at path: {file_path}")

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            raise VideoSourceError(f"Failed to open video file: {file_path}")

        self.is_connected = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self) -> Optional[Frame]:
        if not self.is_connected or not self.cap.isOpened():
            return None

        ret, frame_img = self.cap.read()
        if not ret or frame_img is None:
            logger.info(f"End of file stream reached for camera '{self.camera_id}' ({self.file_path})")
            self.is_connected = False
            return None

        self.frame_count += 1
        
        # Encode frame to JPEG bytes for transient transport if needed
        _, jpeg_buf = cv2.imencode('.jpg', frame_img)
        frame_bytes = jpeg_buf.tobytes()

        metadata = {
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }

        return Frame(
            camera_id=self.camera_id,
            frame_id=self.frame_count,
            timestamp=time.time(),
            frame_bytes=frame_bytes,
            source_type="file",
            source_metadata=metadata
        )

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False


class WebcamVideoSource(VideoSource):
    """Video Source implementation for Local Webcams."""

    def __init__(self, device_index: int = 0, camera_id: str = "CAM-WEBCAM"):
        super().__init__(camera_id=camera_id)
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            logger.warning(f"Webcam device index {device_index} unavailable.")
            self.is_connected = False
        else:
            self.is_connected = True
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self) -> Optional[Frame]:
        if not self.is_connected:
            return None

        ret, frame_img = self.cap.read()
        if not ret or frame_img is None:
            self.is_connected = False
            return None

        self.frame_count += 1
        _, jpeg_buf = cv2.imencode('.jpg', frame_img)

        return Frame(
            camera_id=self.camera_id,
            frame_id=self.frame_count,
            timestamp=time.time(),
            frame_bytes=jpeg_buf.tobytes(),
            source_type="webcam",
            source_metadata={"device_index": self.device_index}
        )

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False


class RTSPVideoSource(VideoSource):
    """Video Source implementation for RTSP Network Streams."""

    def __init__(self, rtsp_url: str, camera_id: str = "CAM-RTSP"):
        super().__init__(camera_id=camera_id)
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

        self.fps = 25.0
        self.total_frames = 500
        if not self.cap.isOpened():
            logger.warning(f"Failed to connect to RTSP stream: {rtsp_url}")
            self.is_connected = False
        else:
            self.is_connected = True
            raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if raw_fps and raw_fps > 0:
                self.fps = raw_fps
            raw_tf = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if raw_tf and raw_tf > 0:
                self.total_frames = raw_tf

    def get_frame(self) -> Optional[Frame]:
        if not self.is_connected:
            return None

        ret, frame_img = self.cap.read()
        if not ret or frame_img is None:
            logger.warning(f"RTSP stream disconnected or frame drop on camera '{self.camera_id}'")
            self.is_connected = False
            return None

        self.frame_count += 1
        _, jpeg_buf = cv2.imencode('.jpg', frame_img)

        return Frame(
            camera_id=self.camera_id,
            frame_id=self.frame_count,
            timestamp=time.time(),
            frame_bytes=jpeg_buf.tobytes(),
            source_type="rtsp",
            source_metadata={"rtsp_url": self.rtsp_url}
        )

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False


class DemoVideoSource(FileVideoSource):
    """Demo Video Source implementation with attached demonstration metadata."""

    def __init__(self, file_path: str, camera_id: str = "CAM-01", demo_name: str = "Normal Patrol"):
        super().__init__(file_path=file_path, camera_id=camera_id)
        self.demo_name = demo_name

    def get_frame(self) -> Optional[Frame]:
        frame = super().get_frame()
        if frame:
            frame.source_type = "demo"
            frame.source_metadata["demo_name"] = self.demo_name
        return frame
