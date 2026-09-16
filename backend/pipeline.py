import time
import logging
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

from backend.interfaces import Frame, AlertOutput, EventPriority, AnalysisFrameResult, CameraStatus
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
from backend.config import settings

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

        # Open-Vocabulary Semantic Discovery & Refinement Engine
        from backend.detection.open_vocab_refiner import OpenVocabEngine
        self.open_vocab_engine = OpenVocabEngine()
        self.last_keyframe_time: float = 0.0

        self.enable_demo_degradation = enable_demo_degradation
        self.last_reliability = None
        self.last_detections = []
        self.last_tracks = []
        self.last_context_events = []
        self.last_alerts = []
        self.last_frame_result: Optional[AnalysisFrameResult] = None

    def get_model_info(self) -> dict:
        """Exposes model diagnostic transparency metadata."""
        info = {}
        if hasattr(self.detector, "get_model_info"):
            info = self.detector.get_model_info()
        if hasattr(self, "open_vocab_engine"):
            info["open_vocab"] = self.open_vocab_engine.get_info()
        return info

    def process_frame(
        self,
        video_source: VideoSource,
        frame_obj: Frame,
        image_np: np.ndarray,
        skip_detection: bool = False
    ) -> Tuple[np.ndarray, List[AlertOutput]]:
        """
        Executes single-frame intelligence pipeline.
        Returns Tuple[annotated_image_np, list_of_new_alerts].
        Backwards-compatible signature, updates self.last_frame_result.
        Supports skip_detection=True for high-speed tracking cadence.
        """
        # Apply programmatic degradation if demo mode active
        if self.enable_demo_degradation:
            image_np = apply_demo_degradation(image_np, blur_ksize=51, brightness_factor=0.3)

        # Buffer frame in pre-event memory ring buffer
        self.ring_buffer.add_frame(frame_obj, image_np)

        # 1. PARALLEL STREAM A: Camera Reliability Engine (Rolling window + Hysteresis)
        rel_score = self.reliability_engine.calculate_reliability(
            camera_id=frame_obj.camera_id,
            frame_id=frame_obj.frame_id,
            timestamp=frame_obj.timestamp,
            image_np=image_np
        )
        self.last_reliability = rel_score

        # 2. Object Detection (YOLO + clean detections)
        if not skip_detection or not hasattr(self, "last_detections") or self.last_detections is None:
            detections = self.detector.detect(frame_obj, image_np=image_np)
            self.last_detections = detections
        else:
            detections = self.last_detections

        # 3. Multi-Object Tracking (ByteTrack / IoU Fallback)
        tracks = self.tracker.update(detections, timestamp=frame_obj.timestamp)

        # 3b. Dual-Path Open-Vocabulary Semantic Discovery & Refinement
        open_vocab_dets = []
        if not skip_detection and getattr(self, "open_vocab_engine", None) and self.open_vocab_engine.is_available:
            # Operation 1: Track Crop Refinement — only refine newly acquired/unconfirmed tracks (max 1/frame)
            if not hasattr(self, "_stabilized_track_ids"):
                self._stabilized_track_ids = set()
            unrefined_tracks = [t for t in tracks if t.track_id not in self._stabilized_track_ids]
            if unrefined_tracks:
                refined = self.open_vocab_engine.refine_track_crops(
                    image_np, [unrefined_tracks[0]],
                    camera_id=frame_obj.camera_id,
                    timestamp=frame_obj.timestamp,
                    frame_id=frame_obj.frame_id
                )
                if refined:
                    open_vocab_dets.extend(refined)
                self._stabilized_track_ids.add(unrefined_tracks[0].track_id)

            # Operation 2: Full-Frame Keyframe Discovery on interval
            interval_sec = getattr(settings, "SMART_DETECTION_INTERVAL_SECONDS", 2.0)
            now_wall = time.time()
            wall_elapsed = now_wall - getattr(self, "_last_full_scan_wall_time", 0.0)
            is_initial = (self.last_keyframe_time == 0.0)
            if is_initial or (wall_elapsed >= max(4.0, interval_sec * 2.0)):
                discovered = self.open_vocab_engine.discover_full_frame(
                    image_np,
                    camera_id=frame_obj.camera_id,
                    timestamp=frame_obj.timestamp,
                    frame_id=frame_obj.frame_id
                )
                if discovered:
                    open_vocab_dets.extend(discovered)
                self.last_keyframe_time = frame_obj.timestamp
                self._last_full_scan_wall_time = now_wall

            if open_vocab_dets:
                tracks = self.tracker.associate_open_vocab_detections(open_vocab_dets, timestamp=frame_obj.timestamp)

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
                active_tracks=tracks,
                reliability_score=rel_score
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

        # Formulate structured AnalysisFrameResult for data contract
        all_active = list(self.alert_manager.active_alerts.values())
        unsupported = getattr(self.detector, "unsupported_classes", ["fence", "stone"])
        self.last_frame_result = AnalysisFrameResult(
            frame_id=frame_obj.frame_id,
            timestamp=frame_obj.timestamp,
            camera_id=frame_obj.camera_id,
            current_detections=detections,
            active_tracks=tracks,
            context_events=ctx_events,
            reliability_score=rel_score,
            new_alerts=new_alerts,
            all_active_alerts=all_active,
            unsupported_classes=unsupported
        )

        # Draw visual tracking overlay
        annotated = ObjectTracker.draw_tracks_overlay(image_np, tracks)

        # Draw scene-level condition banners (fog, camera broken/poor) on top
        adverse_weather_any = any(getattr(e, "adverse_weather", False) for e in ctx_events)
        camera_offline_or_poor = rel_score.status in [CameraStatus.POOR, CameraStatus.OFFLINE]

        annotated = ObjectTracker.draw_scene_overlay(
            annotated,
            adverse_weather=adverse_weather_any,
            camera_broken=camera_offline_or_poor
        )

        return annotated, new_alerts
