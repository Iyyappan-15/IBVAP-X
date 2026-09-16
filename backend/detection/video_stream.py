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
    """Video Source implementation for Local Files and HTTP/HTTPS/RTSP Video Streams."""

    def __init__(self, file_path: str, camera_id: str = "CAM-FILE"):
        super().__init__(camera_id=camera_id)
        self.file_path = file_path
        target_path = file_path

        is_url = file_path.startswith("http://") or file_path.startswith("https://")
        self.is_synthetic = False

        if is_url:
            import hashlib
            import urllib.request
            import ssl
            os.makedirs(settings.VIDEO_TEMP_DIR, exist_ok=True)
            url_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:10]
            cached_filename = f"cached_stream_{camera_id}_{url_hash}.mp4"
            cached_path = os.path.join(settings.VIDEO_TEMP_DIR, cached_filename)

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            if not os.path.exists(cached_path) or os.path.getsize(cached_path) < 1000:
                try:
                    logger.info(f"[VideoSource] Caching public CCTV stream from URL: {file_path}")
                    req = urllib.request.Request(file_path, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                    with urllib.request.urlopen(req, context=ctx, timeout=10) as response, open(cached_path, 'wb') as out_file:
                        out_file.write(response.read())
                    target_path = cached_path
                except Exception as stream_err:
                    logger.warning(f"[VideoSource] Could not stream directly from URL ({stream_err}). Using local fallback asset.")
                    if os.path.exists("data/demo/cctv_night_patrol.mp4"):
                        target_path = "data/demo/cctv_night_patrol.mp4"
                    else:
                        target_path = file_path
            else:
                target_path = cached_path

        if not is_url and not os.path.exists(target_path):
            raise VideoSourceError(f"Video file not found at path: {file_path}")

        self.cap = cv2.VideoCapture(target_path)
        if not self.cap.isOpened() and is_url:
            # Fallback to local asset if video capture failed for URL
            fallback_local = "data/demo/cctv_night_patrol.mp4"
            if os.path.exists(fallback_local):
                self.cap = cv2.VideoCapture(fallback_local)

        if not self.cap.isOpened():
            if not is_url:
                raise VideoSourceError(f"Failed to open video source: {file_path}")
            logger.warning(f"[VideoSource] Could not open video target '{target_path}'. Initializing synthetic sentinel stream fallback.")
            self.is_synthetic = True
            self.is_connected = True
            self.fps = 25.0
            self.width = 640
            self.height = 480
            self.total_frames = 500
        else:
            self.is_connected = True
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            raw_tf = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames = raw_tf if raw_tf > 0 else 500

    def _generate_synthetic_frame(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (30, 25, 20)
        cv2.rectangle(img, (20, 20), (self.width - 20, self.height - 20), (0, 100, 200), 2)
        cv2.line(img, (20, self.height // 2), (self.width - 20, self.height // 2), (0, 255, 0), 1)
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(img, f"SENTINEL FALLBACK FEED | {self.camera_id}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f"Time: {ts_str} | Frame: {self.frame_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return img

    def get_frame(self) -> Optional[Frame]:
        if not self.is_connected:
            return None

        if getattr(self, "is_synthetic", False):
            if self.frame_count >= self.total_frames:
                self.is_connected = False
                return None
            self.frame_count += 1
            frame_img = self._generate_synthetic_frame()
            _, jpeg_buf = cv2.imencode('.jpg', frame_img)
            return Frame(
                camera_id=self.camera_id,
                frame_id=self.frame_count,
                timestamp=time.time(),
                frame_bytes=jpeg_buf.tobytes(),
                source_type="file",
                source_metadata={"file_path": self.file_path, "is_synthetic": True}
            )

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
            "total_frames": self.total_frames,
            "frame_img": frame_img
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
        self.fps = 25.0
        self.total_frames = 500
        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            logger.warning(f"Webcam device index {device_index} unavailable.")
            self.is_connected = False
        else:
            self.is_connected = True
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if raw_fps and raw_fps > 0:
                self.fps = raw_fps

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
