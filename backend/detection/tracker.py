import time
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

from backend.config import settings
from backend.interfaces import Track, Detection, DirectionEnum

logger = logging.getLogger(__name__)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


class PureIoUTracker:
    """Pure NumPy Fallback IoU Tracker when supervision ByteTrack is unavailable."""

    def __init__(self, max_stale_frames: int = 30, iou_threshold: float = 0.3):
        self.max_stale_frames = max_stale_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        matched_track_ids = set()
        matched_det_indices = set()

        # Match existing active tracks with incoming detections by IoU
        for track_id, track_data in list(self.active_tracks.items()):
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue
                iou = calculate_iou(track_data["bbox"], det.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det_idx = idx

            if best_det_idx >= 0:
                matched_det = detections[best_det_idx]
                matched_track_ids.add(track_id)
                matched_det_indices.add(best_det_idx)

                # Update track state
                bbox = matched_det.bbox
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                track_data["bbox"] = bbox
                track_data["trajectory"].append((cx, cy))
                track_data["last_seen"] = timestamp
                track_data["stale_count"] = 0

                # Velocity vector calculation
                if len(track_data["trajectory"]) >= 2:
                    p1 = track_data["trajectory"][-2]
                    p2 = track_data["trajectory"][-1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    track_data["direction_vector"] = (dx, dy)
                    track_data["estimated_speed"] = float(np.sqrt(dx**2 + dy**2))

        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                track_id = self.next_track_id
                self.next_track_id += 1

                bbox = det.bbox
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                self.active_tracks[track_id] = {
                    "track_id": track_id,
                    "class_name": det.class_name,
                    "bbox": bbox,
                    "trajectory": [(cx, cy)],
                    "start_time": timestamp,
                    "last_seen": timestamp,
                    "estimated_speed": 0.0,
                    "direction_vector": (0.0, 0.0),
                    "zone_history": [],
                    "stale_count": 0
                }

        # Track eviction
        for track_id, track_data in list(self.active_tracks.items()):
            if track_id not in matched_track_ids and track_id in self.active_tracks:
                track_data["stale_count"] += 1
                if track_data["stale_count"] > self.max_stale_frames:
                    del self.active_tracks[track_id]

        # Return list of active Track schema instances
        output_tracks = []
        for t_id, data in self.active_tracks.items():
            output_tracks.append(
                Track(
                    track_id=t_id,
                    class_name=data["class_name"],
                    bbox=data["bbox"],
                    trajectory=list(data["trajectory"]),
                    start_time=data["start_time"],
                    last_seen=data["last_seen"],
                    estimated_speed=data["estimated_speed"],
                    direction_vector=data["direction_vector"],
                    direction_enum=DirectionEnum.UNCERTAIN,
                    zone_history=data["zone_history"]
                )
            )
        return output_tracks


class ObjectTracker:
    """Multi-Object Tracker wrapping Supervision ByteTrack with IoU Fallback."""

    def __init__(self, max_stale_frames: int = None):
        self.max_stale_frames = max_stale_frames or settings.TRACK_EVICTION_FRAMES
        self.fallback_tracker = PureIoUTracker(max_stale_frames=self.max_stale_frames)
        self.use_fallback = False

        try:
            import supervision as sv
            self.sv_tracker = sv.ByteTrack(frame_rate=30)
            logger.info("[Tracker] Initialized Supervision ByteTrack multi-object tracker.")
        except Exception as e:
            logger.warning(f"[Tracker] Supervision ByteTrack initialization failed ({e}). Using pure IoU fallback tracker.")
            self.use_fallback = True

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with new detections and return active Track objects."""
        if self.use_fallback:
            return self.fallback_tracker.update(detections, timestamp)

        try:
            import supervision as sv

            if not detections:
                return self.fallback_tracker.update([], timestamp)

            # Convert Detection Pydantic list to supervision Detections object
            xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)

            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

            sv_tracked = self.sv_tracker.update_with_detections(sv_dets)

            # Map ByteTrack output back to our state manager for trajectory maintenance
            mapped_detections = []
            if sv_tracked is not None and len(sv_tracked) > 0:
                for idx in range(len(sv_tracked)):
                    box = sv_tracked.xyxy[idx].tolist()
                    cid = int(sv_tracked.class_id[idx]) if sv_tracked.class_id is not None else 0
                    cname = detections[0].class_name if detections else "object"

                    det = Detection(
                        class_id=cid,
                        class_name=cname,
                        confidence=float(sv_tracked.confidence[idx]) if sv_tracked.confidence is not None else 1.0,
                        bbox=box,
                        camera_id=detections[0].camera_id if detections else "CAM-01",
                        timestamp=timestamp,
                        frame_id=detections[0].frame_id if detections else 1
                    )
                    mapped_detections.append(det)

            return self.fallback_tracker.update(mapped_detections, timestamp)

        except Exception as e:
            logger.warning(f"[Tracker] ByteTrack update failed ({e}). Falling back to IoU tracker.")
            self.use_fallback = True
            return self.fallback_tracker.update(detections, timestamp)

    @staticmethod
    def draw_tracks_overlay(frame_img: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Visual Overlay: Draws bounding boxes, Track IDs, and trajectory polylines on image."""
        annotated = frame_img.copy()

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            
            # Distinct color per track ID
            color_id = (track.track_id * 57) % 255
            color = (int(color_id), int(255 - color_id), 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory polyline
            if len(track.trajectory) >= 2:
                pts = np.array(track.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

            # Draw track ID and class label
            label = f"#{track.track_id} {track.class_name}"
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated
