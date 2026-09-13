"""
Public Camera Source Adapter for IBVAP-X.
Supports HLS (.m3u8), RTSP, MJPEG, and SNAPSHOT (periodic HTTP JPEG image fetcher).
Transparently presents provider metadata and usage modes.
Does NOT mislabel static video files as live HLS feeds.
"""

import time
import logging
import ssl
import hashlib
import os
import urllib.request
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

from backend.interfaces import Frame
from backend.detection.video_stream import VideoSource, VideoSourceError
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class PublicCameraSource(VideoSource):
    """
    Video Source implementation for Public Camera Feeds.
    Supports HLS, RTSP, MJPEG, SNAPSHOT, and VIEW_ONLY modes.
    """

    def __init__(self, camera_info: Dict[str, Any]):
        camera_id = camera_info.get("camera_id", "PUBLIC-CAM-01")
        super().__init__(camera_id=camera_id)

        self.camera_info = camera_info
        self.name = camera_info.get("name", "Public Camera")
        self.country = camera_info.get("country", "")
        self.city = camera_info.get("city", "")
        self.provider = camera_info.get("provider", "Unknown Provider")
        self.source_type = camera_info.get("source_type", "SNAPSHOT").upper()
        self.stream_url = camera_info.get("stream_url", "")
        self.webpage_url = camera_info.get("webpage_url", "")
        self.attribution = camera_info.get("attribution", "")
        self.usage_mode = camera_info.get("usage_mode", "UNKNOWN")
        self.enabled = camera_info.get("enabled", True)

        self.status = "OFFLINE"
        self.last_frame_time = 0.0
        self.stream_fps = 1.0 if self.source_type == "SNAPSHOT" else 25.0
        self.processing_fps = 0.0
        self.width = 640
        self.height = 480
        self.total_frames = 999999
        self.retries = 0
        self.max_retries = settings.PUBLIC_CAMERA_MAX_RETRIES
        self.reconnect_delay = settings.PUBLIC_CAMERA_RECONNECT_SECONDS

        self.cap: Optional[cv2.VideoCapture] = None
        self._last_snapshot_bytes: Optional[bytes] = None
        self._last_snapshot_img: Optional[np.ndarray] = None
        self.snapshot_interval = 3.0  # seconds between periodic fetches for SNAPSHOT

        # SSL context allowing public HTTPS stream certificates
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

        if self.enabled and self.source_type != "VIEW_ONLY" and self.stream_url:
            self._connect()
        elif self.source_type == "VIEW_ONLY":
            self.status = "VIEW_ONLY"

    def _connect(self):
        if not self.stream_url or not self.enabled:
            self.status = "OFFLINE"
            self.is_connected = False
            return

        if self.source_type == "SNAPSHOT":
            img, raw_bytes = self._fetch_snapshot()
            if img is not None:
                self.is_connected = True
                self.status = "ONLINE"
                self.height, self.width = img.shape[:2]
                self._last_snapshot_img = img
                self._last_snapshot_bytes = raw_bytes
                self.last_frame_time = time.time()
                self.retries = 0
            else:
                self.is_connected = False
                self.status = "OFFLINE"
        else:
            # Video stream (HLS, RTSP, MJPEG, STATIC_MP4)
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                ret, frame_img = self.cap.read()
                if ret and frame_img is not None:
                    self.is_connected = True
                    self.status = "ONLINE"
                    self.height, self.width = frame_img.shape[:2]
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        self.stream_fps = fps
                    self.retries = 0
                else:
                    self.cap.release()
                    self.cap = None
                    self._try_url_cache_fallback()
            else:
                self.cap = None
                self._try_url_cache_fallback()

    def _try_url_cache_fallback(self):
        if self.stream_url.startswith("http://") or self.stream_url.startswith("https://"):
            try:
                os.makedirs(settings.VIDEO_TEMP_DIR, exist_ok=True)
                url_hash = hashlib.md5(self.stream_url.encode("utf-8")).hexdigest()[:10]
                ext = ".m3u8" if ".m3u8" in self.stream_url else ".mp4"
                cached_path = os.path.join(
                    settings.VIDEO_TEMP_DIR, f"public_stream_{self.camera_id}_{url_hash}{ext}"
                )

                if not os.path.exists(cached_path) or os.path.getsize(cached_path) < 1000:
                    req = urllib.request.Request(
                        self.stream_url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    )
                    with urllib.request.urlopen(
                        req, context=self._ssl_ctx, timeout=10
                    ) as response, open(cached_path, "wb") as out:
                        out.write(response.read())

                self.cap = cv2.VideoCapture(cached_path)
                if self.cap.isOpened():
                    self.is_connected = True
                    self.status = "ONLINE"
                    self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        self.stream_fps = fps
                    self.retries = 0
                    return
            except Exception as e:
                logger.warning(f"URL stream caching fallback failed for {self.stream_url}: {e}")

        self.is_connected = False
        self.status = "OFFLINE"

    def _fetch_snapshot(self) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
        try:
            req = urllib.request.Request(
                self.stream_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=5) as resp:
                data = resp.read()
                if len(data) > 0:
                    img_np = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img, data
        except Exception as e:
            logger.warning(f"Snapshot fetch error from {self.stream_url}: {e}")
        return None, None

    def get_frame(self) -> Optional[Frame]:
        if self.source_type == "VIEW_ONLY":
            return None

        if not self.is_connected:
            if self.retries < self.max_retries:
                self.retries += 1
                time.sleep(self.reconnect_delay)
                self._connect()
            if not self.is_connected:
                return None

        now = time.time()

        if self.source_type == "SNAPSHOT":
            if now - self.last_frame_time >= self.snapshot_interval or self._last_snapshot_img is None:
                img, raw_bytes = self._fetch_snapshot()
                if img is not None:
                    self._last_snapshot_img = img
                    self._last_snapshot_bytes = raw_bytes
                    self.last_frame_time = now
                    self.retries = 0
                else:
                    self.retries += 1
                    if self.retries >= self.max_retries:
                        self.status = "OFFLINE"
                        self.is_connected = False
                    if self._last_snapshot_img is None:
                        return None

            frame_img = self._last_snapshot_img
            frame_bytes = self._last_snapshot_bytes
        else:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                self.status = "OFFLINE"
                return None

            ret, frame_img = self.cap.read()
            if not ret or frame_img is None:
                # Loop video stream if at end
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame_img = self.cap.read()
                if not ret or frame_img is None:
                    self.is_connected = False
                    self.status = "OFFLINE"
                    return None

            _, jpeg_buf = cv2.imencode(".jpg", frame_img)
            frame_bytes = jpeg_buf.tobytes()

        self.frame_count += 1

        metadata = {
            "camera_id": self.camera_id,
            "name": self.name,
            "provider": self.provider,
            "source_type": self.source_type,
            "country": self.country,
            "city": self.city,
            "attribution": self.attribution,
            "usage_mode": self.usage_mode,
            "webpage_url": self.webpage_url,
            "status": self.status,
            "stream_fps": self.stream_fps,
            "width": self.width,
            "height": self.height,
        }

        return Frame(
            camera_id=self.camera_id,
            frame_id=self.frame_count,
            timestamp=now,
            frame_bytes=frame_bytes,
            source_type="public_camera",
            source_metadata=metadata,
        )

    def release(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        self.is_connected = False
        self.status = "OFFLINE"
