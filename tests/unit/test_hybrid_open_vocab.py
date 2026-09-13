"""
Unit tests for Dual-Path Open-Vocabulary Semantic Engine & Hybrid Detection Architecture.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.config import settings
from backend.interfaces import Detection, Track, DetectionSource
from backend.detection.open_vocab_refiner import OpenVocabEngine
from backend.detection.tracker import ObjectTracker


def test_open_vocab_engine_offline_init():
    """Verifies that OpenVocabEngine handles missing checkpoint gracefully when auto_download=False."""
    engine = OpenVocabEngine(model_path="data/models/non_existent_weights.pt", auto_download=False)
    assert engine.is_available is False
    assert engine.get_info()["is_available"] is False


def test_open_vocab_vocabulary_set_classes():
    """Verifies dynamic class vocabulary configuration."""
    engine = OpenVocabEngine(model_path="data/models/non_existent_weights.pt", auto_download=False)
    prompts = ["person", "dog", "fence", "stone"]
    engine.set_classes(prompts)
    assert engine.active_prompts == ["person", "dog", "fence", "stone"]


@patch("backend.detection.open_vocab_refiner.OpenVocabEngine._initialize_model")
def test_full_frame_discovery_mock(mock_init):
    """Verifies full-frame discovery returns Detection objects with DetectionSource.OPEN_VOCAB_DISCOVERY."""
    engine = OpenVocabEngine(model_path="data/models/yolov8s-worldv2.pt", auto_download=False)
    engine.is_available = True
    engine.active_prompts = ["fence", "stone"]

    # Mock Ultralytics model call
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([100, 100, 200, 300])))]
    mock_box.conf = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0.85))]
    mock_box.cls = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0))]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    engine.model = MagicMock(return_value=[mock_result])
    engine.model.names = {0: "fence", 1: "stone"}

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    discoveries = engine.discover_full_frame(img, prompts=["fence", "stone"], camera_id="CAM-01")

    assert len(discoveries) == 1
    assert discoveries[0].class_name == "fence"
    assert discoveries[0].source == DetectionSource.OPEN_VOCAB_DISCOVERY
    assert discoveries[0].confidence == 0.85


def test_tracker_associate_open_vocab_discovery():
    """Verifies ObjectTracker creates candidate track for unassociated open-vocab discovery."""
    tracker = ObjectTracker()

    det = Detection(
        class_id=99,
        class_name="fence",
        confidence=0.88,
        bbox=[150.0, 150.0, 350.0, 400.0],
        camera_id="CAM-01",
        timestamp=100.0,
        frame_id=1,
        source=DetectionSource.OPEN_VOCAB_DISCOVERY
    )

    tracks = tracker.associate_open_vocab_detections([det], timestamp=100.0)

    assert len(tracks) >= 1
    fence_track = next((t for t in tracks if t.class_name == "fence"), None)
    assert fence_track is not None
    assert fence_track.source == DetectionSource.OPEN_VOCAB_DISCOVERY
    assert fence_track.label_stability == "HIGH"


def test_tracker_label_stabilization():
    """Verifies temporal confidence-weighted label stabilization on rolling track trajectory."""
    tracker = ObjectTracker()

    # Track data dict
    tdata = {
        "track_id": 1,
        "class_name": "person",
        "bbox": [10.0, 10.0, 50.0, 100.0],
        "class_history": [("person", 0.90), ("person", 0.85), ("dog", 0.30)],
        "label_stability": "HIGH"
    }

    tracker._update_track_label_stability(tdata)

    assert tdata["class_name"] == "person"
    assert tdata["label_stability"] == "HIGH"

    # Confused history with heavy low-confidence noise
    tdata["class_history"] = [("person", 0.40), ("dog", 0.35), ("car", 0.35)]
    tracker._update_track_label_stability(tdata)
    assert tdata["label_stability"] in ["MEDIUM", "LOW"]
