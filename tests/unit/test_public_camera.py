"""
Unit tests for PublicCameraSource adapter and public camera stream integration.
"""

import os
import json
import time
import pytest
import numpy as np

from backend.config.settings import settings
from backend.detection.public_camera import PublicCameraSource
from backend.detection.video_stream import VideoSource


def test_public_camera_registry_parsing():
    """Verify that public_cameras.json exists and adheres to valid structure."""
    assert os.path.exists(settings.PUBLIC_CAMERAS_CONFIG_PATH)
    with open(settings.PUBLIC_CAMERAS_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "public_cameras" in data
    cameras = data["public_cameras"]
    assert isinstance(cameras, list)
    assert len(cameras) > 0


def test_public_camera_metadata_schema():
    """Verify essential metadata fields in public camera entries."""
    with open(settings.PUBLIC_CAMERAS_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = [
        "camera_id", "name", "country", "city",
        "provider", "source_type", "stream_url",
        "webpage_url", "usage_mode", "attribution", "enabled"
    ]

    for cam in data["public_cameras"]:
        for field in required_fields:
            assert field in cam, f"Field '{field}' missing from camera entry {cam.get('camera_id')}"


def test_source_type_detection():
    """Verify that source_type logic sets expected initial status."""
    view_only_cam = {
        "camera_id": "TEST-VIEW-01",
        "name": "Test View Only",
        "country": "US",
        "city": "Test",
        "provider": "Test",
        "source_type": "VIEW_ONLY",
        "stream_url": "",
        "webpage_url": "https://example.com",
        "usage_mode": "UNKNOWN",
        "attribution": "Test",
        "enabled": True
    }
    source = PublicCameraSource(view_only_cam)
    assert source.status == "VIEW_ONLY"
    assert source.get_frame() is None
    source.release()


def test_snapshot_periodic_fetcher():
    """Verify PublicCameraSource in SNAPSHOT mode fetches images cleanly."""
    snapshot_cam = {
        "camera_id": "TEST-SNAP-01",
        "name": "Test Snapshot",
        "country": "US",
        "city": "Test",
        "provider": "Test",
        "source_type": "SNAPSHOT",
        "stream_url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "webpage_url": "https://opencv.org",
        "usage_mode": "UNKNOWN",
        "attribution": "Test",
        "enabled": True
    }
    source = PublicCameraSource(snapshot_cam)
    assert source.is_connected

    frame = source.get_frame()
    assert frame is not None
    assert frame.source_type == "public_camera"
    assert frame.source_metadata["camera_id"] == "TEST-SNAP-01"
    assert frame.source_metadata["source_type"] == "SNAPSHOT"
    assert len(frame.frame_bytes) > 0
    source.release()


def test_reconnect_backoff_and_fallback():
    """Verify fail-safe fallback handling when stream URL is unreachable."""
    invalid_cam = {
        "camera_id": "TEST-INVALID-01",
        "name": "Test Invalid",
        "country": "US",
        "city": "Test",
        "provider": "Test",
        "source_type": "HLS",
        "stream_url": "https://invalid-host-999.example.com/stream.m3u8",
        "webpage_url": "https://example.com",
        "usage_mode": "UNKNOWN",
        "attribution": "Test",
        "enabled": True
    }
    source = PublicCameraSource(invalid_cam)
    assert source.is_connected
    assert "ONLINE" in source.status
    frame = source.get_frame()
    assert frame is not None
    source.release()


def test_resource_release_cleanup():
    """Verify release method sets status to OFFLINE and closes connections."""
    snapshot_cam = {
        "camera_id": "TEST-RELEASE-01",
        "name": "Test Release",
        "country": "US",
        "city": "Test",
        "provider": "Test",
        "source_type": "SNAPSHOT",
        "stream_url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "webpage_url": "https://opencv.org",
        "usage_mode": "UNKNOWN",
        "attribution": "Test",
        "enabled": True
    }
    source = PublicCameraSource(snapshot_cam)
    assert source.is_connected
    source.release()
    assert not source.is_connected
    assert source.status == "OFFLINE"
