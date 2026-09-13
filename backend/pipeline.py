import time
import logging
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

from backend.interfaces import Frame, AlertOutput, EventPriority
from backend.detection.video_stream import VideoSource, FileVideoSource
from backend.detection.detector import ObjectDetector
from backend.detection.tracker import ObjectTracker
from backend.context.context_engine import ContextEngine
from backend.reliability.reliability_score import CameraReliabilityEngine, apply_demo_degradation
from backend.anomaly.detector import AnomalyDetector
from backend.correlation.event_correlator import EventCorrelator
from backend.alerts.alert_manager import AlertManager
from backend.evidence.buffer import PreEventRingBuffer
from backend.evidence.capture import EvidenceCapturer

logger = logging.getLogger(__name__)

class IBVAPXPipeline:
    """
    Core Independent IBVAP-X Intelligence Pipeline.
    Completely decoupled from Streamlit and FastAPI — both act as clients.
    """

    def __init__(self, config_path: str = "data/config/cameras.json", enable_demo_degradation: bool = False):
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.context_engine = ContextEngine(config_path=config_path)
        self.reliability_engine = CameraReliabilityEngine()
        self.anomaly_detector = AnomalyDetector()
        self.correlator = EventCorrelator(config_path=config_path)
        self.alert_manager = AlertManager()
        self.evidence_capturer = EvidenceCapturer()
        self.ring_buffer = PreEventRingBuffer(max_seconds=10, fps=10.0)

        self.enable_demo_degradation = enable_demo_degradation
        self.last_reliability = None
        self.last_detections = []
        self.last_tracks = []
        self.last_context_events = []
        self.last_alerts = []

    def process_frame(
        self,
        video_source: VideoSource,
        frame_obj: Frame,
        image_np: np.ndarray
    ) -> Tuple[np.ndarray, List[AlertOutput]]:
        """
        Executes single-frame intelligence pipeline.
        Returns Tuple[annotated_image_np, list_of_new_alerts].
        """
        # Apply programmatic degradation if demo mode active
        if self.enable_demo_degradation:
            image_np = apply_demo_degradation(image_np, blur_ksize=51, brightness_factor=0.3)

        # Buffer frame in pre-event memory ring buffer
        self.ring_buffer.add_frame(frame_obj, image_np)

        # 1. PARALLEL STREAM A: Camera Reliability Engine
        rel_score = self.reliability_engine.calculate_reliability(
            camera_id=frame_obj.camera_id,
            frame_id=frame_obj.frame_id,
            timestamp=frame_obj.timestamp,
            image_np=image_np
        )
        self.last_reliability = rel_score

        # 2. Object Detection (with Sub-Box Containment Suppression)
        detections = self.detector.detect(frame_obj, image_np=image_np)
        self.last_detections = detections

        # 3. Multi-Object Tracking (ByteTrack / IoU Fallback)
        tracks = self.tracker.update(detections, timestamp=frame_obj.timestamp)
        self.last_tracks = tracks

        new_alerts: List[AlertOutput] = []
        ctx_events = []

        # Process each active track through parallel engines
        for track in tracks:
            # 4. PARALLEL STREAM B: Context Engine
            ctx_event = self.context_engine.analyze_track(
                camera_id=frame_obj.camera_id,
                track=track,
                frame_width=image_np.shape[1],
                frame_height=image_np.shape[0],
                timestamp=frame_obj.timestamp,
                image_np=image_np,
                active_tracks=tracks
            )
            ctx_events.append(ctx_event)

            # 5. PARALLEL STREAM C: Anomaly Engine (Optional, non-blocking)
            anomaly_res = self.anomaly_detector.evaluate_track(
                camera_id=frame_obj.camera_id,
                track=track
            )
            anom_score = anomaly_res.normalized_anomaly_score if anomaly_res else None

            # 6. Cross-Camera Correlation Check (Evidence Contributor)
            corr_event = self.correlator.process_event(ctx_event)
            is_cross_confirmed = corr_event is not None

            # 7. Priority Engine & Alert Manager Processing
            alert = self.alert_manager.process_event(
                context_event=ctx_event,
                reliability_score=rel_score,
                cross_camera_confirmed=is_cross_confirmed,
                anomaly_score=anom_score
            )

            if alert:
                # 8. Evidence Auto-Capture for High/Critical Alerts
                if alert.event_priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
                    pre_frames = self.ring_buffer.get_buffered_frames()
                    self.evidence_capturer.capture_evidence(
                        alert=alert,
                        pre_event_frames=pre_frames,
                        current_frame_img=image_np
                    )
                new_alerts.append(alert)

        self.last_context_events = ctx_events
        self.last_alerts = new_alerts

        # Draw visual tracking overlay
        annotated = ObjectTracker.draw_tracks_overlay(image_np, tracks)

        return annotated, new_alerts
