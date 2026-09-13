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
from backend.interfaces import Track, Detection, DirectionEnum, DetectionSource

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
                            "class_history": [(cname, 0.85)],
                            "label_stability": "HIGH",
                            "source": DetectionSource.YOLO
                        }
                    else:
                        tdata = self.active_tracks[tid]
                        tdata["bbox"] = box
                        tdata["last_seen"] = timestamp
                        tdata["stale_count"] = 0
                        # Record class observation for label stabilization
                        tdata["class_history"].append((cname, 0.85))
                        if len(tdata["class_history"]) > 10:
                            tdata["class_history"].pop(0)
                        self._update_track_label_stability(tdata)

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

    def associate_open_vocab_detections(self, open_vocab_dets: List[Detection], timestamp: float) -> List[Track]:
        """
        Associates open-vocabulary discovery and refinement detections with active tracks,
        or creates new discovery candidate tracks for unsupported objects (fence, stone, gate).
        """
        if not open_vocab_dets:
            return self._build_track_outputs()

        next_id = max(self.active_tracks.keys(), default=0) + 100

        for det in open_vocab_dets:
            best_track_id = None
            best_iou = 0.0

            for tid, tdata in self.active_tracks.items():
                iou = calculate_iou(tdata["bbox"], det.bbox)
                if iou > best_iou and iou >= 0.25:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                # Associated with existing track
                tdata = self.active_tracks[best_track_id]
                # High weight for open-vocab refinement / discovery
                weight = 1.5 if det.source == DetectionSource.OPEN_VOCAB_REFINEMENT else 1.2
                tdata["class_history"].append((det.class_name, det.confidence * weight))
                if len(tdata["class_history"]) > 10:
                    tdata["class_history"].pop(0)
                tdata["source"] = det.source
                self._update_track_label_stability(tdata)
            elif det.source == DetectionSource.OPEN_VOCAB_DISCOVERY:
                # Instantiate new open-vocab discovery candidate track
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0
                self.active_tracks[next_id] = {
                    "track_id": next_id,
                    "class_name": det.class_name,
                    "bbox": det.bbox,
                    "trajectory": [(cx, cy)],
                    "start_time": timestamp,
                    "last_seen": timestamp,
                    "estimated_speed": 0.0,
                    "direction_vector": (0.0, 0.0),
                    "zone_history": [],
                    "stale_count": 0,
                    "class_history": [(det.class_name, det.confidence * 1.5)],
                    "label_stability": "HIGH",
                    "source": DetectionSource.OPEN_VOCAB_DISCOVERY
                }
                next_id += 1

        return self._build_track_outputs()

    def _update_track_label_stability(self, tdata: Dict):
        """Calculates rolling confidence-weighted class label and stability score."""
        class_hist = tdata.get("class_history", [])
        if not class_hist:
            return

        scores: Dict[str, float] = {}
        total_weight = 0.0
        for cname, conf in class_hist:
            scores[cname] = scores.get(cname, 0.0) + conf
            total_weight += conf

        if total_weight <= 0.0:
            return

        top_class = max(scores.items(), key=lambda x: x[1])
        top_weight_ratio = top_class[1] / total_weight

        assigned_class = top_class[0]

        # Biomechanical sanity check: A human track standing/walking has height > width (AR >= 1.35)
        # Ground-level quadruped animals have horizontal/compact proportions (AR <= 1.25)
        bbox = tdata.get("bbox", [0, 0, 0, 0])
        tbw = max(1.0, float(bbox[2] - bbox[0]))
        tbh = max(1.0, float(bbox[3] - bbox[1]))
        if assigned_class == "person" and (tbh / tbw) <= 1.25 and tbh < 220.0:
            assigned_class = "dog"

        # Human vs handheld item protection: An upright tall object is NEVER a bottle, cup, or stone
        if assigned_class in ("bottle", "cup", "stone", "knife", "cell phone"):
            if tbh >= 70.0 and (tbh / tbw) >= 1.30:
                assigned_class = "person"
            elif any(c == "person" for c, _ in class_hist):
                assigned_class = "person"

        tdata["class_name"] = assigned_class
        if top_weight_ratio >= 0.70:
            tdata["label_stability"] = "HIGH"
        elif top_weight_ratio >= 0.40:
            tdata["label_stability"] = "MEDIUM"
        else:
            tdata["label_stability"] = "LOW"

    def _build_track_outputs(self) -> List[Track]:
        # Track-Level Containment Suppression: Evict nested partial sub-boxes (e.g. torso/arm inside full body)
        active_items = list(self.active_tracks.items())

        def box_area(b):
            return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))

        # Sort by box area descending so larger bounding boxes take precedence
        active_items.sort(key=lambda item: box_area(item[1]["bbox"]), reverse=True)
        suppressed_ids = set()

        for i, (tid_b, data_b) in enumerate(active_items):
            if tid_b in suppressed_ids:
                continue
            box_b = data_b["bbox"]
            area_b = box_area(box_b)
            if area_b <= 0:
                continue

            for j in range(i):
                tid_a, data_a = active_items[j]
                if tid_a in suppressed_ids:
                    continue

                box_a = data_a["bbox"]
                area_a = box_area(box_a)
                if area_a <= area_b:
                    continue

                ix1 = max(box_a[0], box_b[0])
                iy1 = max(box_a[1], box_b[1])
                ix2 = min(box_a[2], box_b[2])
                iy2 = min(box_a[3], box_b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                containment = inter / area_b

                # Containment check for same class (especially person)
                if data_a["class_name"] == data_b["class_name"]:
                    # If box B is 30%+ contained inside larger box A of same class, evict B
                    if containment >= 0.30:
                        suppressed_ids.add(tid_b)
                        break

                # Suppress false handheld items / small items that cover a person's torso
                if data_a["class_name"] == "person" and data_b["class_name"] in ("bottle", "stone", "cup", "cell phone", "knife"):
                    if containment >= 0.35 and area_b > 0.08 * area_a:
                        suppressed_ids.add(tid_b)
                        break

        for sid in suppressed_ids:
            if sid in self.active_tracks:
                del self.active_tracks[sid]

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
                    label_stability=data.get("label_stability", "HIGH"),
                    source=data.get("source", DetectionSource.YOLO)
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
            "person": (0, 140, 255),          # Orange
            "car": (255, 180, 0),             # Cyan/Amber
            "dog": (0, 215, 255),             # Golden Yellow
            "cat": (0, 235, 255),             # Bright Yellow
            "bottle": (255, 140, 0),          # Sky Blue
            "fence": (0, 230, 70),            # Bright Neon Green
            "fence_perimeter": (0, 230, 70),  # Bright Neon Green
            "stone": (180, 0, 255),           # Purple/Violet
            "knife": (0, 0, 230),             # Red
            "weapon": (0, 0, 200),            # Dark Red
            "truck": (255, 100, 0),           # Deep Blue
            "bus": (200, 200, 0),             # Cyan
            "motorcycle": (0, 220, 255),      # Yellow
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

    @staticmethod
    def draw_scene_overlay(
        frame_img: np.ndarray,
        adverse_weather: bool = False,
        camera_broken: bool = False
    ) -> np.ndarray:
        """
        Draws scene-level condition banners on the frame:
        - ⛅ FOG / LOW VISIBILITY banner (amber) when adverse_weather is True
        - 🔴 CAMERA DESTROYED banner (red flashing) when camera_broken is True

        Call this AFTER draw_tracks_overlay so it appears on top.
        """
        annotated = frame_img.copy()
        h, w = annotated.shape[:2]
        banner_y = 10

        if camera_broken:
            # Bold full-width red banner at top
            banner_h = 34
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            label = "!! CAMERA DESTROYED — CRITICAL ALERT — DISPATCH IMMEDIATELY !!"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            tx = max(4, (w - tw) // 2)
            cv2.putText(annotated, label, (tx, banner_y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            banner_y += banner_h + 6

        if adverse_weather:
            # Amber banner below the broken camera banner (if any)
            banner_h = 28
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 160, 220), -1)
            cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
            label = "⚠  FOG / LOW VISIBILITY DETECTED — REDUCED DETECTION ACCURACY"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            tx = max(4, (w - tw) // 2)
            cv2.putText(annotated, label, (tx, banner_y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
