import pytest
import os
import cv2
import numpy as np

from backend.detection.video_stream import FileVideoSource
from backend.detection.detector import ObjectDetector
from backend.detection.tracker import ObjectTracker
from backend.context.context_engine import ContextEngine
from backend.interfaces import ContextEvent

def test_milestone_a_pipeline_end_to_end():
    """
    Milestone A Gate Integration Test:
    Video Ingestion → YOLO Detection → ByteTrack Tracking → Context Engine → Context Event.
    Executes end-to-end on synthetic/uploaded test video without mock data.
    """
    video_path = os.path.join("data", "demo", "test_zone.mp4")
    assert os.path.exists(video_path), "Test video 'test_zone.mp4' must exist"

    # 1. Video Ingestion
    source = FileVideoSource(file_path=video_path, camera_id="CAM-01")
    assert source.is_connected is True

    # 2. Object Detection
    model_path = os.path.join("data", "models", "yolov8n.pt")
    detector = ObjectDetector(model_path=model_path, confidence_threshold=0.20)

    # 3. Object Tracking
    tracker = ObjectTracker()

    # 4. Context Engine
    context_engine = ContextEngine(config_path="data/config/cameras.json")

    processed_frames = 0
    generated_context_events = []

    # Stream frames through pipeline
    while source.is_connected and processed_frames < 30:
        frame_obj = source.get_frame()
        if frame_obj is None:
            break

        processed_frames += 1
        
        # Decode frame image
        nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        detections = detector.detect(frame_obj, image_np=img)

        # Run Multi-Object Tracker
        tracks = tracker.update(detections, timestamp=frame_obj.timestamp)

        # Process Context Engine for active tracks
        for track in tracks:
            context_evt = context_engine.analyze_track(
                camera_id=frame_obj.camera_id,
                track=track,
                frame_width=source.width,
                frame_height=source.height,
                timestamp=frame_obj.timestamp
            )
            generated_context_events.append(context_evt)

    source.release()

    assert processed_frames > 0, "Pipeline must process frames"
    print(f"[Milestone A] Processed {processed_frames} frames, generated {len(generated_context_events)} context events.")
    
    # Verify schema of generated context events
    for evt in generated_context_events:
        assert isinstance(evt, ContextEvent)
        assert evt.camera_id == "CAM-01"
        assert isinstance(evt.in_restricted_zone, bool)
