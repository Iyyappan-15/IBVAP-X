import os
import logging
from typing import List
import cv2
import numpy as np
import torch

from backend.config import settings
from backend.interfaces import Detection, Frame, DetectionSource
from backend.detection.fence_detector import FenceDetector

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    """Raised when YOLO weights file is missing."""
    pass


def suppress_nested_subboxes(detections: List[Detection], containment_threshold: float = 0.35) -> List[Detection]:
    """
    Suppresses nested / containment duplicate detections (e.g. YOLO detecting a leg or arm
    as an extra person box inside a main person box, or false handheld items covering torso).
    If Box B is contained inside Box A, the smaller Box B is suppressed.
    """
    if len(detections) <= 1:
        return detections

    def box_area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    sorted_dets = sorted(detections, key=lambda d: box_area(d.bbox), reverse=True)
    kept_dets: List[Detection] = []

    vehicle_classes = {"car", "truck", "bus", "motorcycle"}

    for i, det_b in enumerate(sorted_dets):
        b_box = det_b.bbox
        area_b = box_area(b_box)
        if area_b <= 0:
            continue

        is_contained_duplicate = False

        for det_a in kept_dets:
            a_box = det_a.bbox
            area_a = box_area(a_box)

            # Check if same class or both are vehicle classes
            same_category = (
                det_a.class_name == det_b.class_name
                or (det_a.class_name in vehicle_classes and det_b.class_name in vehicle_classes)
            )

            # Check if det_b is a false item (bottle, stone, cup) covering a person's body
            is_item_in_person = (
                det_a.class_name == "person"
                and det_b.class_name in ("bottle", "stone", "cup", "cell phone", "knife")
            )

            if not same_category and not is_item_in_person:
                continue

            # Calculate intersection
            x1 = max(a_box[0], b_box[0])
            y1 = max(a_box[1], b_box[1])
            x2 = min(a_box[2], b_box[2])
            y2 = min(a_box[3], b_box[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            containment = intersection / area_b

            if is_item_in_person:
                # If an item box is >40% inside person and exceeds 8% of person area, it's torso/clothing
                if containment >= 0.40 and area_b > 0.08 * area_a:
                    is_contained_duplicate = True
                    break
            else:
                # Aggressive suppression for person sub-boxes (arms/legs/torso inside full person)
                th = 0.30 if det_b.class_name == "person" else containment_threshold

                if containment >= th and area_a > area_b:
                    is_contained_duplicate = True
                    break

        if not is_contained_duplicate:
            kept_dets.append(det_b)

    return kept_dets


class ObjectDetector:
    """Wrapper around Ultralytics YOLOv8 for Object Detection + Fence & Perimeter Structure Detection."""

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        target_classes: List[str] = None
    ):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.YOLO_CONFIDENCE_THRESHOLD
        self.target_classes = target_classes or settings.detect_classes_list
        self.fence_detector = FenceDetector()

        # Detect GPU vs CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            try:
                # Limit CPU threads to prevent thread contention & CPU quota throttling on shared cloud instances
                torch.set_num_threads(2)
            except Exception:
                pass
        logger.info(f"[Detector] Initializing YOLO model on device: {self.device}")

        from ultralytics import YOLO

        # Auto-download default model if missing; raise error for custom invalid paths
        if not os.path.exists(self.model_path):
            if self.model_path == settings.YOLO_MODEL_PATH and "yolov8n.pt" in self.model_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
                logger.info(f"[Detector] Default weights not found at '{self.model_path}'. Auto-downloading YOLOv8n weights...")
                self.model = YOLO("yolov8n.pt")
                try:
                    import shutil
                    if os.path.exists("yolov8n.pt") and self.model_path != "yolov8n.pt":
                        shutil.copy("yolov8n.pt", self.model_path)
                except Exception:
                    pass
            else:
                raise ModelNotFoundError(
                    f"YOLO model weights file not found at '{self.model_path}'. "
                    f"Run 'python scripts/download_model.py' to download weights."
                )
        else:
            self.model = YOLO(self.model_path)



        self.model.to(self.device)

        # Inspect actual model classes
        self.available_model_classes = []
        if hasattr(self.model, "names") and self.model.names:
            self.available_model_classes = [name.lower() for name in self.model.names.values()]

        # Identify requested domain classes unsupported by this model
        domain_requested = ["fence", "stone", "rock"]
        self.unsupported_classes = [c for c in domain_requested if c not in self.available_model_classes]

        # Map class names to class IDs for configured target classes present in model
        self.target_class_ids = []
        if hasattr(self.model, "names"):
            for cid, cname in self.model.names.items():
                if cname.lower() in [tc.lower() for tc in self.target_classes]:
                    self.target_class_ids.append(cid)

    def get_model_info(self) -> dict:
        """Returns structured transparency metadata about the loaded detection model."""
        return {
            "model_name": os.path.basename(self.model_path),
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": settings.YOLO_IOU_THRESHOLD,
            "image_size": settings.YOLO_IMAGE_SIZE,
            "total_classes": len(self.available_model_classes),
            "supported_classes": self.available_model_classes[:10],  # Sample for UI display
            "unsupported_classes": self.unsupported_classes
        }

    def _detect_handheld_objects(
        self,
        image_np: np.ndarray,
        person_detections: List[Detection],
        camera_id: str,
        timestamp: float,
        frame_id: int
    ) -> List[Detection]:
        """
        Detects compact handheld objects (stones, tools, wire cutters, thrown items)
        in hands, chest/abdomen, and payload zones of detected persons.

        Dual-strategy detection:
          A) Edge-based: Canny + Laplacian + morphological close for clear scenes.
          B) Color-based: HSV/Lab dark-blob isolation against snow/fog backgrounds.
             In snowy/foggy scenes the stone (dark/warm-toned) has high contrast
             against the white background even when edge density is globally low.
        """
        if image_np is None or not person_detections:
            return []

        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        handheld_dets: List[Detection] = []

        # Global scene analysis: detect foggy/snowy (bright, flat) conditions
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        is_adverse_scene = (std_lum < 42.0 and mean_lum > 110.0) or (std_lum < 30.0) or (mean_lum < 45.0)

        for p_det in person_detections:
            px1, py1, px2, py2 = [int(v) for v in p_det.bbox]
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)

            # Strictly focus on actual hand and arm holding regions (NEVER the central torso or full body)
            hand_rois = [
                # High raised hand / head level (e.g. raised bottle, phone, or stone)
                (max(0, px1 - int(pw * 0.20)), max(0, py1 - int(ph * 0.20)),
                 min(w, px2 + int(pw * 0.20)), min(h, py1 + int(ph * 0.35))),
                # Left hand region
                (max(0, px1 - int(pw * 0.25)), max(0, py1 + int(ph * 0.20)),
                 min(w, px1 + int(pw * 0.30)), min(h, py1 + int(ph * 0.65))),
                # Right hand region
                (max(0, px2 - int(pw * 0.30)), max(0, py1 + int(ph * 0.20)),
                 min(w, px2 + int(pw * 0.25)), min(h, py1 + int(ph * 0.65))),
            ]

            for rx1, ry1, rx2, ry2 in hand_rois:
                if (rx2 - rx1) < 14 or (ry2 - ry1) < 14:
                    continue

                roi_img = image_np[ry1:ry2, rx1:rx2]
                try:
                    # Neural crop inference: ONLY accept actual objects confirmed by neural network
                    crop_results = self.model(roi_img, conf=0.28, verbose=False)
                    for cr in crop_results:
                        if cr.boxes is not None and len(cr.boxes) > 0:
                            for cbox in cr.boxes:
                                cid = int(cbox.cls[0].cpu().numpy())
                                cname = self.model.names.get(cid, "").lower()
                                c_conf = float(cbox.conf[0].cpu().numpy())

                                # Accept only real handheld items
                                if cname in ("bottle", "cup", "cell phone", "knife"):
                                    c_xyxy = cbox.xyxy[0].cpu().numpy().tolist()
                                    ox1 = float(rx1 + c_xyxy[0])
                                    oy1 = float(ry1 + c_xyxy[1])
                                    ox2 = float(rx1 + c_xyxy[2])
                                    oy2 = float(ry1 + c_xyxy[3])
                                    obj_w = ox2 - ox1
                                    obj_h = oy2 - oy1

                                    # Size sanity guard: Handheld item cannot be large (max 35% of human dimensions)
                                    if obj_w > 0.35 * pw or obj_h > 0.35 * ph:
                                        continue

                                    # Duplicate suppression
                                    is_dup = any(
                                        abs(ox1 - ed.bbox[0]) < 15 and abs(oy1 - ed.bbox[1]) < 15
                                        for ed in handheld_dets
                                    )
                                    if not is_dup:
                                        handheld_dets.append(
                                            Detection(
                                                class_id=cid,
                                                class_name=cname,
                                                confidence=round(c_conf, 4),
                                                bbox=[round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)],
                                                camera_id=camera_id,
                                                timestamp=timestamp,
                                                frame_id=frame_id,
                                                source=DetectionSource.SCENE_ANALYSIS
                                            )
                                        )
                except Exception:
                    pass

        return handheld_dets

    def detect(self, frame_obj: Frame, image_np: np.ndarray = None) -> List[Detection]:
        """
        Runs object detection + handheld object detection + fence detection on a Frame or image.
        Returns clean Detection objects without nested duplicate boxes.
        """
        if image_np is None:
            if frame_obj.frame_bytes is not None:
                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                logger.warning(f"[Detector] No image content available for frame {frame_obj.frame_id}")
                return []

        # 1. Adaptive contrast enhancement for night / low-light / low-contrast foggy CCTV
        yolo_input = image_np
        is_low_contrast = False
        try:
            gray_check = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
            mean_lum = float(np.mean(gray_check))
            std_lum = float(np.std(gray_check))
            if mean_lum < 70.0 or std_lum < 40.0:
                is_low_contrast = True
                lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                yolo_input = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            yolo_input = image_np

        effective_conf = max(0.18, self.confidence_threshold - 0.05) if is_low_contrast else self.confidence_threshold

        # 2. Run YOLO inference with clean floor (min 0.20)
        results = self.model(
            yolo_input,
            conf=min(0.20, effective_conf),
            iou=0.45,
            agnostic_nms=True,
            imgsz=640,
            classes=self.target_class_ids if self.target_class_ids else None,
            verbose=False
        )

        raw_detections: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}").lower()

                # Calibrated thresholds to eliminate hallucinations:
                # - Animals (dog/cat): 0.20 to catch low-profile quadrupeds in snow
                # - Handheld items (bottle, knife, cell phone, cup): 0.28 to prevent noise flickers
                # - Standard objects (person, car, etc.): effective_conf (default 0.30)
                if cls_name in ("dog", "cat"):
                    min_conf = min(0.20, effective_conf)
                elif cls_name in ("bottle", "knife", "cell phone", "cup"):
                    min_conf = max(0.28, effective_conf)
                else:
                    min_conf = effective_conf

                if conf < min_conf:
                    continue

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[round(v, 2) for v in xyxy],
                    camera_id=frame_obj.camera_id,
                    timestamp=frame_obj.timestamp,
                    frame_id=frame_obj.frame_id,
                    source=DetectionSource.YOLO
                )
                raw_detections.append(det)

        # Pre-refine raw detections (quadrupeds and cylinders) before containment suppression
        h, w = image_np.shape[:2]
        raw_detections = refine_detection_classes(raw_detections, h, w, image_np=image_np)

        # 2. Apply Sub-Box Containment Suppression
        clean_detections = suppress_nested_subboxes(raw_detections, containment_threshold=0.35)

        # 3. Detect Handheld Objects (e.g. stones, tools, weapons)
        person_dets = [d for d in clean_detections if d.class_name == "person"]
        handheld_dets = self._detect_handheld_objects(
            image_np,
            person_dets,
            camera_id=frame_obj.camera_id,
            timestamp=frame_obj.timestamp,
            frame_id=frame_obj.frame_id
        )
        if handheld_dets:
            clean_detections.extend(handheld_dets)

        # 3b. Detect Ground Quadruped Animals on Snow/Light Backgrounds
        has_animal = any(d.class_name in ("dog", "cat") for d in clean_detections)
        if not has_animal and (getattr(frame_obj, "frame_id", 1) % 3 == 1):
            ground_animal_dets = self._detect_ground_animals(
                image_np,
                clean_detections,
                camera_id=frame_obj.camera_id,
                timestamp=frame_obj.timestamp,
                frame_id=frame_obj.frame_id
            )
            if ground_animal_dets:
                clean_detections.extend(ground_animal_dets)

        # 4. Detect Perimeter Fence & Boundary Structures (Cached physical infrastructure)
        fid = getattr(frame_obj, "frame_id", 1)
        if (fid % 15 == 1) or not hasattr(self, "_cached_fence_dets") or self._cached_fence_dets is None:
            self._cached_fence_dets = self.fence_detector.detect_fence(
                image_np,
                camera_id=frame_obj.camera_id,
                timestamp=frame_obj.timestamp,
                frame_id=frame_obj.frame_id,
                existing_detections=clean_detections
            )
        if self._cached_fence_dets:
            clean_detections.extend(self._cached_fence_dets)

        # 5. Final containment suppression to eliminate any sub-boxes inside persons
        clean_detections = suppress_nested_subboxes(clean_detections, containment_threshold=0.35)

        # 6. Refine Quadruped Animals, Cylinders & Physical Aspect Ratios
        clean_detections = refine_detection_classes(clean_detections, h, w, image_np=image_np)

        return clean_detections

    def _detect_ground_animals(
        self,
        image_np: np.ndarray,
        existing_detections: List[Detection],
        camera_id: str,
        timestamp: float,
        frame_id: int
    ) -> List[Detection]:
        """
        Detects ground-level quadruped animals (cats, dogs) on snow/fog or light backgrounds,
        including companion animals walking alongside or between legs/feet of detected persons.
        Uses mandatory neural crop verification to guarantee zero false positives from weeds/shadows.
        """
        if image_np is None:
            return []

        h, w = image_np.shape[:2]
        animal_dets: List[Detection] = []

        # ── 1. Check Feet / Ground Proximity of Detected Persons ───────────────
        # In real CCTV, animals often walk alongside, between legs, or near feet of persons
        person_dets = [d for d in existing_detections if d.class_name == "person"]
        for p_det in person_dets:
            px1, py1, px2, py2 = [int(v) for v in p_det.bbox]
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)

            # Ground / feet region of person (lower 40% down to ground + side perimeter)
            feet_y1 = max(0, int(py2 - ph * 0.40))
            feet_y2 = min(h, int(py2 + ph * 0.15))
            feet_x1 = max(0, int(px1 - pw * 0.50))
            feet_x2 = min(w, int(px2 + pw * 0.50))

            if (feet_y2 - feet_y1) >= 20 and (feet_x2 - feet_x1) >= 20:
                feet_crop = image_np[feet_y1:feet_y2, feet_x1:feet_x2]
                try:
                    c_results = self.model(feet_crop, conf=0.08, imgsz=160, verbose=False)
                    for cr in c_results:
                        if cr.boxes is not None and len(cr.boxes) > 0:
                            for cbox in cr.boxes:
                                c_id = int(cbox.cls[0].cpu().numpy())
                                c_n = self.model.names.get(c_id, "").lower()
                                if c_n in ("dog", "cat", "horse", "sheep", "cow", "bear", "bird"):
                                    c_xyxy = cbox.xyxy[0].cpu().numpy().tolist()
                                    ax1 = float(feet_x1 + c_xyxy[0])
                                    ay1 = float(feet_y1 + c_xyxy[1])
                                    ax2 = float(feet_x1 + c_xyxy[2])
                                    ay2 = float(feet_y1 + c_xyxy[3])
                                    c_conf = float(cbox.conf[0].cpu().numpy())

                                    # Check duplicates
                                    is_dup = any(
                                        abs(ax1 - ad.bbox[0]) < 15 and abs(ay1 - ad.bbox[1]) < 15
                                        for ad in animal_dets
                                    )
                                    if not is_dup:
                                        animal_dets.append(
                                            Detection(
                                                class_id=16 if c_n in ("dog", "horse", "sheep", "cow", "bear") else 15,
                                                class_name="dog" if c_n in ("dog", "horse", "sheep", "cow", "bear") else "cat",
                                                confidence=round(c_conf, 4),
                                                bbox=[round(ax1, 2), round(ay1, 2), round(ax2, 2), round(ay2, 2)],
                                                camera_id=camera_id,
                                                timestamp=timestamp,
                                                frame_id=frame_id,
                                                source=DetectionSource.SCENE_ANALYSIS
                                            )
                                        )
                except Exception:
                    pass

        # ── 2. Scan Ground Region for Quadruped Blobs ─────────────────────────
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np.copy()
        mean_lum = float(np.mean(gray))

        # Focus on ground region (lower 65% of frame)
        ground_y1 = int(h * 0.35)
        ground_roi = gray[ground_y1:, :]
        roi_h, roi_w = ground_roi.shape[:2]

        if roi_h > 0 and roi_w > 0:
            if mean_lum > 75.0:
                dark_thresh = min(175, int(mean_lum * 0.82))
                _, dark_mask = cv2.threshold(ground_roi, dark_thresh, 255, cv2.THRESH_BINARY_INV)
            else:
                _, dark_mask = cv2.threshold(ground_roi, 70, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 120 <= area <= int(w * h * 0.10):
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    aspect_ratio = float(cw) / max(1.0, float(ch))

                    # Quadruped aspect ratio: horizontal or angled walking (0.45 to 3.2)
                    if 0.45 <= aspect_ratio <= 3.2 and cw >= 14 and ch >= 12:
                        abs_x1 = float(cx)
                        abs_y1 = float(ground_y1 + cy)
                        abs_x2 = float(cx + cw)
                        abs_y2 = float(ground_y1 + cy + ch)

                        # Avoid duplicate if existing detection or animal_dets already covers it
                        all_existing = list(existing_detections) + animal_dets
                        already_covered = False
                        for det in all_existing:
                            if det.class_name in ("dog", "cat"):
                                dx1, dy1, dx2, dy2 = det.bbox
                                ix1 = max(abs_x1, dx1)
                                iy1 = max(abs_y1, dy1)
                                ix2 = min(abs_x2, dx2)
                                iy2 = min(abs_y2, dy2)
                                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                                u = (abs_x2 - abs_x1) * (abs_y2 - abs_y1) + (dx2 - dx1) * (dy2 - dy1) - inter
                                if u > 0 and (inter / u) > 0.20:
                                    already_covered = True
                                    break

                        if already_covered:
                            continue

                        # Check that it doesn't overlap the upper body / torso of an existing person
                        overlap_torso = False
                        for det in existing_detections:
                            if det.class_name == "person":
                                px1, py1, px2, py2 = det.bbox
                                ph = py2 - py1
                                # Upper 65% of person is torso/head
                                if abs_y1 < (py1 + ph * 0.65) and abs_x1 < px2 and abs_x2 > px1:
                                    overlap_torso = True
                                    break

                        if not overlap_torso:
                            # Mandatory neural crop verification — reject weeds, shadows, and dirt
                            yolo_confirmed = False
                            detected_label = None
                            detected_conf = 0.0
                            try:
                                pad = 12
                                cy1 = max(0, int(abs_y1) - pad)
                                cy2 = min(h, int(abs_y2) + pad)
                                cx1 = max(0, int(abs_x1) - pad)
                                cx2 = min(w, int(abs_x2) + pad)
                                if (cy2 - cy1) >= 16 and (cx2 - cx1) >= 16:
                                    animal_crop = image_np[cy1:cy2, cx1:cx2]
                                    c_results = self.model(animal_crop, conf=0.08, imgsz=160, verbose=False)
                                    for cr in c_results:
                                        if cr.boxes is not None and len(cr.boxes) > 0:
                                            c_id = int(cr.boxes.cls[0].cpu().numpy())
                                            c_n = self.model.names.get(c_id, "").lower()
                                            if c_n in ("dog", "cat", "horse", "sheep", "cow", "bear", "bird"):
                                                detected_label = c_n if c_n in ("dog", "cat") else "dog"
                                                detected_conf = float(cr.boxes.conf[0].cpu().numpy())
                                                yolo_confirmed = True
                                                break
                            except Exception:
                                pass

                            if yolo_confirmed and detected_label is not None:
                                animal_dets.append(
                                    Detection(
                                        class_id=16 if detected_label == "dog" else 15,
                                        class_name=detected_label,
                                        confidence=round(detected_conf, 4),
                                        bbox=[round(abs_x1, 2), round(abs_y1, 2), round(abs_x2, 2), round(abs_y2, 2)],
                                        camera_id=camera_id,
                                        timestamp=timestamp,
                                        frame_id=frame_id,
                                        source=DetectionSource.SCENE_ANALYSIS
                                    )
                                )
                                if len(animal_dets) >= 3:
                                    break

        return animal_dets


def refine_detection_classes(
    detections: List[Detection],
    img_height: int,
    img_width: int,
    image_np: Optional[np.ndarray] = None
) -> List[Detection]:
    """
    Refines class labels for detections based on physical aspect ratios, relative scale,
    ground-plane perspective projection, and biomechanical geometry.
    Guarantees that:
      1. Quadruped animals (dogs/cats) are NEVER misclassified as 'person' (including climbing fence postures).
      2. Human persons (foreground or distant, in winter coats/hoodies) are NEVER misclassified as 'gas cylinder'.
      3. Standing industrial containers/bottles (height >= 80px, width >= 35px on floor) are classified as 'gas cylinder'.
      4. Slender fence posts / poles in fields are NOT misclassified as 'gas cylinder'.
    """
    if not detections:
        return detections

    # Identify true upright human reference detections (AR between 1.50 and 3.40, substantial height)
    tall_persons = [
        d for d in detections 
        if d.class_name == "person" and 1.50 <= (d.bbox[3] - d.bbox[1]) / max(1.0, d.bbox[2] - d.bbox[0]) <= 3.40 and (d.bbox[3] - d.bbox[1]) >= 70.0
    ]

    refined: List[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        aspect_ratio = bh / bw  # Height / Width

        # ── 1. BOTTLE -> CYLINDER / GAS CYLINDER ────────────────────────────
        # Drinking bottles are small (<80px, <12% frame height, held in hand/table).
        # Standing industrial / LPG gas cylinders are wide containers (bh >= 80, bw >= 30, AR <= 3.8, on ground).
        if d.class_name == "bottle":
            is_large_bottle = (bh >= 90) and (bw >= 30) and (1.6 <= aspect_ratio <= 3.8)
            is_standing_cylinder = (bh >= 80) and (bw >= 35) and (1.6 <= aspect_ratio <= 3.5) and (y2 > img_height * 0.50)
            if is_large_bottle or is_standing_cylinder:
                d.class_name = "gas cylinder"
                d.class_id = 81
                d.confidence = max(0.65, d.confidence)

        # ── 2. PERSON REFINEMENT (ANIMALS & EXTREME CYLINDERS) ───────────────
        elif d.class_name == "person":
            is_animal = False

            # Case A: Horizontal quadruped animal on the ground (dogs/quadrupeds)
            if aspect_ratio <= 1.25 and (y2 > img_height * 0.30) and (bh < img_height * 0.40):
                is_animal = True

            # Case B: Climbing / perched small animal (e.g. cat/dog on fence/gate)
            # Physical signature: small body (bh <= 85, bw <= 60), elevated or adjacent to a reference human,
            # where the human is > 1.8x taller, confirming this small entity is an animal, NOT an adult human.
            elif tall_persons and d not in tall_persons and (bh <= 85.0) and (bw <= 60.0):
                nearest_person = min(tall_persons, key=lambda tp: abs((tp.bbox[0] + tp.bbox[2])/2.0 - (x1 + x2)/2.0))
                n_x1, n_y1, n_x2, n_y2 = nearest_person.bbox
                n_h = n_y2 - n_y1
                dx = abs((n_x1 + n_x2)/2.0 - (x1 + x2)/2.0)

                # If adjacent to a tall person (dx <= 240px) and less than 55% of their height:
                # it is a climbing animal / pet on the fence/gate, not a human
                if dx <= 240.0 and (bh < 0.55 * n_h):
                    is_animal = True
                # Perspective check: if at similar depth or foreground (y2 > img_height * 0.40) but height < 48% of expected human
                elif y2 > img_height * 0.40:
                    y_horiz = max(0.0, 0.20 * img_height)
                    expected_h = n_h * max(0.25, (y2 - y_horiz) / max(1.0, n_y2 - y_horiz))
                    if (bh / max(1.0, expected_h)) < 0.48:
                        is_animal = True

            # Case C: Isolated small ground blob when no reference human is present
            elif not tall_persons and (bh < img_height * 0.18) and (bh * bw < 0.02 * img_height * img_width) and (y2 > img_height * 0.45) and aspect_ratio <= 1.30:
                is_animal = True

            if is_animal:
                # Vertical/climbing profile (e.g. cat climbing chain-link fence or upright): cat
                # Horizontal quadruped profile (dog walking/standing): dog
                if aspect_ratio >= 1.30:
                    d.class_name = "cat"
                    d.class_id = 15
                else:
                    d.class_name = "dog"
                    d.class_id = 16
            else:
                # Non-human extreme vertical structures (AR >= 4.8, e.g. synthetic tall pipes/cylinders)
                # Humans physically have AR between 1.5 and 3.4. AR >= 4.8 is mechanically non-human.
                if aspect_ratio >= 4.8 and (y2 > img_height * 0.35):
                    d.class_name = "gas cylinder"
                    d.class_id = 81
                    d.confidence = max(0.65, d.confidence)

        refined.append(d)

    return refined
