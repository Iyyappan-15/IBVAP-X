"""
backend/upload/safe_temp_storage.py

Handles safe temporary file creation for uploaded videos.
- Uses uuid-based filenames (no path traversal possible)
- Ensures the temp directory exists
- Provides cleanup helper (does NOT delete evidence from HIGH/CRITICAL events)
"""
from __future__ import annotations

import os
import uuid
import logging
import shutil

logger = logging.getLogger(__name__)


def get_temp_dir(base_dir: str = "data/uploads_temp") -> str:
    """Return the absolute temp directory path, creating it if needed."""
    os.makedirs(base_dir, exist_ok=True)
    return os.path.abspath(base_dir)


def save_upload_to_temp(
    file_bytes: bytes,
    original_filename: str,
    base_dir: str = "data/uploads_temp",
) -> str:
    """
    Write raw upload bytes to a safe temp file on disk.

    Returns the absolute path to the saved file.
    The filename is <uuid>.<original_ext> — no user-supplied name is used
    in the path, preventing path traversal.
    """
    ext = _safe_ext(original_filename)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    temp_dir = get_temp_dir(base_dir)
    dest_path = os.path.join(temp_dir, safe_name)

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved upload to temp: %s (%d bytes)", dest_path, len(file_bytes))
    return dest_path


def cleanup_temp_file(file_path: str) -> None:
    """
    Delete a temporary file if it exists.
    NEVER call this on evidence output directories — only temp upload files.
    """
    try:
        if file_path and os.path.isfile(file_path):
            os.remove(file_path)
            logger.info("Cleaned up temp file: %s", file_path)
    except OSError as e:
        logger.warning("Could not delete temp file %s: %s", file_path, e)


def cleanup_old_uploads(base_dir: str = "data/uploads_temp", max_age_hours: int = 24) -> None:
    """
    Remove temp upload files older than max_age_hours.
    Called at startup to prevent disk accumulation.
    """
    import time

    temp_dir = get_temp_dir(base_dir)
    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for fname in os.listdir(temp_dir):
        fpath = os.path.join(temp_dir, fname)
        if os.path.isfile(fpath):
            try:
                mtime = os.path.getmtime(fpath)
                if mtime < cutoff:
                    os.remove(fpath)
                    logger.info("Removed stale temp upload: %s", fpath)
            except OSError:
                pass


def _safe_ext(filename: str) -> str:
    """Extract and validate extension; default to .mp4 if unknown."""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    return ext if ext in {".mp4", ".avi", ".mov"} else ".mp4"
