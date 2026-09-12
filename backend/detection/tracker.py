"""
backend/detection/tracker.py

Multi-Object Tracking: Supervision ByteTrack with Pure IoU fallback.

Key fixes:
1. Proper ByteTrack ID persistence directly from sv_tracked.tracker_id.
2. Trajectory jump filtering: Disallows teleportation lines across the screen when IDs switch.
3. Stationary object filter: Parked cars (<15px movement) do NOT draw stray trajectory lines.
4. Class-isolated IoU matching in FallbackTracker: A car track NEVER matches a person detection.
5. Smooth anti-aliased fading trail for moving targets (last 30 points).
"""
import logging
from typing import List, Dict, Tuple, Optional
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
# Pure IoU Fallback Tracker (Class-Isolated)
# ─────────────────────────────────────────────────────────────────────────────

class PureIoUTracker:
    """Pure NumPy fallback tracker used when Supervision ByteTrack is unavailable."""

    def __init__(self, max_stale_frames: int = 30, iou_threshold: float = 0.35):
        self.max_stale_frames = max_stale_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.active_tracks: Dict[int, Dict] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        matched_track_ids: set = set()
        matched_det_indices: set = set()

        # ── Match existing tracks to incoming detections by IoU + SAME CLASS ──
        for track_id, track_data in list(self.active_tracks.items()):
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_det_indices:
                    continue
                # Class isolation: car can only match car, person only person
                if det.class_name != track_data["class_name"]:
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
                track_data["last_seen"] = timestamp
                track_data["stale_count"] = 0

                # Append to trajectory with teleportation guard
                traj = track_data["trajectory"]
                if traj:
                    last_pt = traj[-1]
                    dist = float(np.hypot(cx - last_pt[0], cy - last_pt[1]))
                    if dist > 80.0:
                        # Teleportation jump detected — reset trajectory to prevent streak lines
                        track_data["trajectory"] = [(cx, cy)]
                    elif dist >= 2.0:
                        # Meaningful movement — append
                        track_data["trajectory"].append((cx, cy))
                else:
                    track_data["trajectory"].append((cx, cy))

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
# Main ObjectTracker (ByteTrack Direct State Management)
# ─────────────────────────────────────────────────────────────────────────────

class ObjectTracker:
    """Multi-Object Tracker: Supervision ByteTrack with Direct State & Pure IoU fallback."""

    def __init__(self, max_stale_frames: int = None):
        self.max_stale_frames = max_stale_frames or settings.TRACK_EVICTION_FRAMES
        self.fallback_tracker = PureIoUTracker(max_stale_frames=self.max_stale_frames)
        self.active_tracks: Dict[int, Dict] = {}
        self.use_fallback = False

        try:
            import supervision as sv
            self.sv_tracker = sv.ByteTrack(frame_rate=30)
            logger.info("[Tracker] Supervision ByteTrack initialised.")
        except Exception as exc:
            logger.warning("[Tracker] ByteTrack unavailable (%s). Using IoU fallback.", exc)
            self.use_fallback = True

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        """Update tracker with current-frame detections; return active Track list."""
        if self.use_fallback:
            return self.fallback_tracker.update(detections, timestamp)

        try:
            import supervision as sv

            if not detections:
                # Evict stale tracks
                for t_id in list(self.active_tracks.keys()):
                    self.active_tracks[t_id]["stale_count"] += 1
                    if self.active_tracks[t_id]["stale_count"] > self.max_stale_frames:
                        del self.active_tracks[t_id]
                return self._build_track_outputs()

            cid_to_name: Dict[int, str] = {d.class_id: d.class_name for d in detections}

            xyxy       = np.array([d.bbox       for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id   = np.array([d.class_id   for d in detections], dtype=int)

            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            sv_tracked = self.sv_tracker.update_with_detections(sv_dets)

            seen_ids = set()

            if sv_tracked is not None and len(sv_tracked) > 0 and sv_tracked.tracker_id is not None:
                for idx in range(len(sv_tracked)):
                    tid = int(sv_tracked.tracker_id[idx])
                    box = [round(float(v), 2) for v in sv_tracked.xyxy[idx].tolist()]
                    cid = int(sv_tracked.class_id[idx]) if sv_tracked.class_id is not None else 0
                    cname = cid_to_name.get(cid, detections[0].class_name if detections else "object")

                    seen_ids.add(tid)
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0

                    if tid not in self.active_tracks:
                        self.active_tracks[tid] = {
                            "track_id": tid,
                            "class_name": cname,
                            "bbox": box,
                            "trajectory": [(cx, cy)],
                            "start_time": timestamp,
                            "last_seen": timestamp,
                            "estimated_speed": 0.0,
                            "direction_vector": (0.0, 0.0),
                            "zone_history": [],
                            "stale_count": 0,
                        }
                    else:
                        tdata = self.active_tracks[tid]
                        tdata["bbox"] = box
                        tdata["class_name"] = cname
                        tdata["last_seen"] = timestamp
                        tdata["stale_count"] = 0

                        traj = tdata["trajectory"]
                        if traj:
                            last_pt = traj[-1]
                            dist = float(np.hypot(cx - last_pt[0], cy - last_pt[1]))
                            if dist > 80.0:
                                # Teleportation jump guard
                                tdata["trajectory"] = [(cx, cy)]
                            elif dist >= 1.5:
                                tdata["trajectory"].append((cx, cy))
                        else:
                            tdata["trajectory"].append((cx, cy))

                        if len(tdata["trajectory"]) >= 2:
                            p1 = tdata["trajectory"][-2]
                            p2 = tdata["trajectory"][-1]
                            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                            tdata["direction_vector"] = (dx, dy)
                            tdata["estimated_speed"] = float(np.sqrt(dx ** 2 + dy ** 2))

            # Evict tracks not seen this frame
            for t_id in list(self.active_tracks.keys()):
                if t_id not in seen_ids:
                    self.active_tracks[t_id]["stale_count"] += 1
                    if self.active_tracks[t_id]["stale_count"] > self.max_stale_frames:
                        del self.active_tracks[t_id]

            return self._build_track_outputs()

        except Exception as exc:
            logger.warning("[Tracker] ByteTrack update failed (%s). Switching to IoU fallback.", exc)
            self.use_fallback = True
            return self.fallback_tracker.update(detections, timestamp)

    def _build_track_outputs(self) -> List[Track]:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Visual overlay with Jump & Stationary Protection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_tracks_overlay(frame_img: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw bounding boxes, clean badges, and motion-only fading trajectory lines.

        Clean Rendering Rules:
        - Only moving objects (total displacement > 15px) have trajectory lines drawn.
        - Stationary objects (parked cars) only show bounding box + label badge (no lines).
        - No lines drawn across point gaps > 50px (jump/occlusion guard).
        """
        annotated = frame_img.copy()

        # Class-specific theme colours (BGR)
        CLASS_COLORS = {
            "person": (0, 140, 255),       # Orange
            "car": (255, 180, 0),          # Cyan/Amber
            "truck": (255, 100, 0),        # Deep Blue
            "bus": (200, 200, 0),          # Cyan
            "motorcycle": (0, 220, 255),   # Yellow
        }

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]

            # Choose distinctive color
            color = CLASS_COLORS.get(track.class_name.lower(), (0, 255, 128))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Trajectory — ONLY for moving objects
            traj = track.trajectory[-35:]
            if len(traj) >= 2:
                # Check total displacement from start of window to end
                total_disp = float(np.hypot(traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1]))

                # Only draw trail if object has actually moved at least 15 pixels
                if total_disp >= 15.0:
                    n = len(traj)
                    for i in range(1, n):
                        pt1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
                        pt2 = (int(traj[i][0]),     int(traj[i][1]))
                        seg_dist = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

                        # Jump guard: do not draw line if points jumped > 50 pixels
                        if seg_dist > 50.0:
                            continue

                        # Fade alpha from 0.30 (oldest) → 1.0 (newest)
                        alpha = 0.30 + 0.70 * (i / n)
                        seg_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated, pt1, pt2, seg_color, 2, cv2.LINE_AA)

            # Label badge
            label = f"#{track.track_id} {track.class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx, ly = x1, max(th + 6, y1 - 6)
            cv2.rectangle(annotated, (lx, ly - th - 5), (lx + tw + 6, ly + 3), color, -1)
            cv2.putText(
                annotated, label,
                (lx + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated
