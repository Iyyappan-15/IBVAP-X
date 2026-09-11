"""
backend/upload/video_validator.py

Full video validation pipeline for the IBVAP-X upload addendum.
Validates extension, readability, file size, duration, resolution,
FPS, and that at least one frame is decodable.

All limits are read from settings so they can be overridden in .env.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2

from backend.config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
ALLOWED_EXTENSIONS_DISPLAY = "MP4 (H.264) — Recommended, AVI, MOV"


@dataclass
class ValidationResult:
    """Outcome of a full video validation check."""
    passed: bool = False
    # Errors block processing
    errors: list[str] = field(default_factory=list)
    # Warnings are non-fatal — shown to user but do not block
    warnings: list[str] = field(default_factory=list)
    # Populated metadata (available even when validation partially fails)
    filename: str = ""
    file_size_mb: float = 0.0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    codec_fourcc: str = ""
    estimated_process_time_seconds: float = 0.0

    @property
    def error_message(self) -> str:
        return " | ".join(self.errors)

    @property
    def summary_label(self) -> str:
        if self.passed:
            return "Video passed all validation checks."
        return f"Validation FAILED: {self.error_message}"


def validate_upload(file_path: str, original_filename: str = "") -> ValidationResult:
    """
    Run all validation steps on a video file that has already been saved to disk.

    Steps (in order — stops early on fatal errors):
    1. Extension check
    2. File size check
    3. OpenCV readability check
    4. Duration check (min / max)
    5. Resolution check (with warnings for <720p or >1080p)
    6. FPS sanity check
    7. At-least-one-frame decode test

    Returns a ValidationResult.  Does NOT raise exceptions — all errors
    are captured inside the result so the caller can display them cleanly.
    """
    result = ValidationResult(filename=original_filename or os.path.basename(file_path))

    # ── 1. Extension ────────────────────────────────────────────────────
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        result.errors.append(
            f"Unsupported file type '{ext}'. "
            f"Supported: {ALLOWED_EXTENSIONS_DISPLAY}"
        )
        return result   # Cannot proceed without a valid extension

    # ── 2. File size ─────────────────────────────────────────────────────
    try:
        size_bytes = os.path.getsize(file_path)
        result.file_size_mb = size_bytes / (1024 * 1024)
    except OSError as e:
        result.errors.append(f"Cannot read file: {e}")
        return result

    if result.file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        result.errors.append(
            f"File size {result.file_size_mb:.1f} MB exceeds the "
            f"{settings.MAX_UPLOAD_SIZE_MB} MB upload limit."
        )
        return result

    # ── 3. OpenCV readability ─────────────────────────────────────────────
    try:
        cap = cv2.VideoCapture(file_path)
    except Exception as e:
        result.errors.append(f"OpenCV failed to open file: {e}")
        return result

    if not cap.isOpened():
        result.errors.append(
            "Video file could not be opened by the decoder. "
            "Try re-encoding to MP4 (H.264)."
        )
        cap.release()
        return result

    # ── 4. Metadata extraction ─────────────────────────────────────────────
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    result.fps = raw_fps if raw_fps and raw_fps > 0 else 25.0
    result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        result.codec_fourcc = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip()
    except Exception:
        result.codec_fourcc = "Unknown"

    if result.fps > 0 and result.total_frames > 0:
        result.duration_seconds = result.total_frames / result.fps
    else:
        result.duration_seconds = 0.0

    # ── 5. Duration check ─────────────────────────────────────────────────
    if result.duration_seconds < settings.MIN_VIDEO_DURATION_SECONDS:
        result.warnings.append(
            f"Video is very short ({result.duration_seconds:.1f}s). "
            f"Minimum recommended: {settings.MIN_VIDEO_DURATION_SECONDS}s."
        )

    if result.duration_seconds > settings.MAX_VIDEO_DURATION_SECONDS:
        result.errors.append(
            f"Video duration {result.duration_seconds:.1f}s exceeds the "
            f"{settings.MAX_VIDEO_DURATION_SECONDS}s limit. "
            "Trim the video before uploading."
        )
        cap.release()
        return result

    # ── 6. Resolution check ───────────────────────────────────────────────
    if result.width == 0 or result.height == 0:
        result.errors.append("Could not determine video resolution.")
        cap.release()
        return result

    if result.width < 640 or result.height < 480:
        result.warnings.append(
            f"Resolution {result.width}x{result.height} is below 480p. "
            "Detection quality may be poor."
        )
    elif result.width > 1920 or result.height > 1080:
        result.warnings.append(
            f"Resolution {result.width}x{result.height} exceeds 1080p. "
            "Frames will be resized for inference only "
            f"(original evidence preserved at full resolution)."
        )
    elif result.width < 1280 or result.height < 720:
        result.warnings.append(
            f"Resolution {result.width}x{result.height} is below recommended 720p. "
            "For best detection, use 1280x720 or higher."
        )

    # ── 7. FPS sanity check ───────────────────────────────────────────────
    if raw_fps <= 0 or raw_fps > 120:
        result.warnings.append(
            f"Unusual FPS reported by decoder ({raw_fps}). "
            "Processing will use 25 FPS as fallback."
        )

    # ── 8. At-least-one-frame decode test ─────────────────────────────────
    ok, test_frame = cap.read()
    cap.release()

    if not ok or test_frame is None:
        result.errors.append(
            "Video opened but no frames could be decoded. "
            "The file may be corrupted or use an unsupported codec. "
            "Re-encode to MP4 H.264 and try again."
        )
        return result

    # ── Compute estimated processing time ─────────────────────────────────
    effective_process_fps = min(settings.PROCESS_FPS, result.fps)
    frames_to_process = int(result.duration_seconds * effective_process_fps)
    # Rough estimate: ~0.15 s per frame on CPU with YOLOv8n
    result.estimated_process_time_seconds = frames_to_process * 0.15

    result.passed = True
    logger.info(
        "Video validation passed: %s (%.1f MB, %.1fs, %dx%d @ %.1f fps)",
        result.filename,
        result.file_size_mb,
        result.duration_seconds,
        result.width,
        result.height,
        result.fps,
    )
    return result


def resize_for_inference(
    image_np,
    max_width: int | None = None,
    max_height: int | None = None,
):
    """
    Resize a frame for inference only if it exceeds the configured max dimensions.
    Does NOT upscale.  Returns (resized_image, was_resized).
    """
    import numpy as np  # local import to avoid top-level dep if not needed

    mw = max_width or settings.MAX_INFERENCE_WIDTH
    mh = max_height or settings.MAX_INFERENCE_HEIGHT

    h, w = image_np.shape[:2]
    if w <= mw and h <= mh:
        return image_np, False

    scale = min(mw / w, mh / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True
