"""
backend/detection/tracker.py

Multi-Object Tracking: Supervision ByteTrack with Pure IoU fallback.

Bug fixes applied:
- Task 3: Each ByteTrack output box now gets the correct class_name via
  class_id → name lookup built from the current frame's detections.
  Previously all boxes wrongly received detections[0].class_name, causing
  a single car to be labelled as 'person' or vice-versa.
- Task 4: draw_tracks_overlay caps trajectory to last 60 points and fades
  older segments so long diagonal lines no longer span the entire frame.
"""
import logging
from typing import List, Dict

import numpy as np
import cv2

from backend.config import settings
from backend.interfaces import Track, Detection, DirectionEnum

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pure IoU Fallback Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PureIoUTracker:
    """Pure NumPy fallback tracker used when Supervision ByteTrack is unavailable."""

    def __init__(self, max_stale_frames: int = 30, iou_threshold: float = 0.3):
        self.max_stale_frames = max_stale_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        matched_track_ids: set = set()
        matched_det_indices: set = set()

        # ── Match existing tracks to incoming detections by IoU ────────────
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

                bbox = matched_det.bbox
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                track_data["bbox"] = bbox
                track_data["class_name"] = matched_det.class_name   # keep class current
                track_data["trajectory"].append((cx, cy))
                track_data["last_seen"] = timestamp
                track_data["stale_count"] = 0

                if len(track_data["trajectory"]) >= 2:
                    p1 = track_data["trajectory"][-2]
                    p2 = track_data["trajectory"][-1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    track_data["direction_vector"] = (dx, dy)
                    track_data["estimated_speed"] = float(np.sqrt(dx ** 2 + dy ** 2))

        # ── Create new tracks for unmatched detections ─────────────────────
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
                    "stale_count": 0,
                }

        # ── Evict stale tracks ─────────────────────────────────────────────
        for track_id, track_data in list(self.active_tracks.items()):
            if track_id not in matched_track_ids:
                track_data["stale_count"] += 1
                if track_data["stale_count"] > self.max_stale_frames:
                    del self.active_tracks[track_id]

        # ── Return active Track schema instances ───────────────────────────
        output: List[Track] = []
        for t_id, data in self.active_tracks.items():
            output.append(
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
                    zone_history=data["zone_history"],
                )
            )
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Main ObjectTracker (ByteTrack + IoU fallback)
# ─────────────────────────────────────────────────────────────────────────────

class ObjectTracker:
    """Multi-Object Tracker: Supervision ByteTrack with Pure IoU fallback."""

    def __init__(self, max_stale_frames: int = None):
        self.max_stale_frames = max_stale_frames or settings.TRACK_EVICTION_FRAMES
        # Always instantiate fallback first (used for empty frames & as state store)
        self.fallback_tracker = PureIoUTracker(max_stale_frames=self.max_stale_frames)
        self.use_fallback = False

        try:
            import supervision as sv  # noqa: F401
            self.sv_tracker = sv.ByteTrack(frame_rate=30)
            logger.info("[Tracker] Supervision ByteTrack initialised.")
        except Exception as exc:
            logger.warning(
                "[Tracker] ByteTrack unavailable (%s). Using IoU fallback.", exc
            )
            self.use_fallback = True

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with current-frame detections; return active Track list."""
        if self.use_fallback:
            return self.fallback_tracker.update(detections, timestamp)

        try:
            import supervision as sv

            if not detections:
                return self.fallback_tracker.update([], timestamp)

            # ── Build class_id → class_name map for THIS frame ──────────────
            # Task 3 fix: every tracked box receives the class_name that
            # matches its own class_id, not always detections[0].class_name.
            cid_to_name: Dict[int, str] = {d.class_id: d.class_name for d in detections}

            xyxy       = np.array([d.bbox       for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id   = np.array([d.class_id   for d in detections], dtype=int)

            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            sv_tracked = self.sv_tracker.update_with_detections(sv_dets)

            mapped: List[Detection] = []
            if sv_tracked is not None and len(sv_tracked) > 0:
                for idx in range(len(sv_tracked)):
                    box  = sv_tracked.xyxy[idx].tolist()
                    cid  = int(sv_tracked.class_id[idx]) if sv_tracked.class_id is not None else 0
                    # Correct class name for this specific class_id
                    cname = cid_to_name.get(cid, detections[0].class_name)
                    mapped.append(
                        Detection(
                            class_id=cid,
                            class_name=cname,
                            confidence=(
                                float(sv_tracked.confidence[idx])
                                if sv_tracked.confidence is not None else 1.0
                            ),
                            bbox=box,
                            camera_id=detections[0].camera_id,
                            timestamp=timestamp,
                            frame_id=detections[0].frame_id,
                        )
                    )

            return self.fallback_tracker.update(mapped, timestamp)

        except Exception as exc:
            logger.warning("[Tracker] ByteTrack update failed (%s). Switching to IoU.", exc)
            self.use_fallback = True
            return self.fallback_tracker.update(detections, timestamp)

    # ─────────────────────────────────────────────────────────────────────────
    # Visual overlay
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_tracks_overlay(frame_img: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw bounding boxes, labels, and fading trajectory lines on a frame.

        Task 4 fix — trajectory rendering:
        - Only the last 60 trajectory points are drawn (prevents the long
          diagonal line that spans the entire frame when an object has been
          tracked for many frames).
        - Older segments are drawn at ~35 % brightness; the line fades up
          to full brightness at the current position.
        """
        annotated = frame_img.copy()

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            # Unique BGR colour per track ID
            hue = (track.track_id * 57) % 255
            color = (int(hue), int(255 - hue), 255)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # ── Trajectory — capped at last 60 points, fading ────────────
            traj = track.trajectory[-60:]   # <-- KEY FIX: cap trajectory length
            n = len(traj)
            if n >= 2:
                for i in range(1, n):
                    # alpha: 0.35 for oldest segment → 1.0 for newest
                    alpha = 0.35 + 0.65 * (i / n)
                    seg_color = tuple(int(c * alpha) for c in color)
                    pt1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
                    pt2 = (int(traj[i][0]),     int(traj[i][1]))
                    cv2.line(annotated, pt1, pt2, seg_color, 2, cv2.LINE_AA)

            # Label: track ID + class name + confidence indicator
            label = f"#{track.track_id} {track.class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx, ly = x1, max(th + 4, y1 - 4)
            # Filled label background for readability
            cv2.rectangle(annotated, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(
                annotated, label,
                (lx + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated
