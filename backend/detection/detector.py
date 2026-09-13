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


def suppress_nested_subboxes(detections: List[Detection], containment_threshold: float = 0.40) -> List[Detection]:
    """
    Suppresses nested / containment duplicate detections (e.g. YOLO detecting a leg or arm
    as an extra person box inside a main person box, or car window inside a car).
    If Box B is contained inside Box A of the same category, the smaller Box B is suppressed.
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

            if not same_category:
                continue

            # Calculate intersection
            x1 = max(a_box[0], b_box[0])
            y1 = max(a_box[1], b_box[1])
            x2 = min(a_box[2], b_box[2])
            y2 = min(a_box[3], b_box[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            containment = intersection / area_b

            # Aggressive suppression for person sub-boxes (arms/legs/torso inside full person)
            th = 0.35 if det_b.class_name == "person" else containment_threshold

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

        # Precompute Lab + HSV for color-based method
        lab_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        hsv_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

        for p_det in person_detections:
            px1, py1, px2, py2 = [int(v) for v in p_det.bbox]
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)

            # ROIs covering all holding positions (including raised hands near head/shoulder)
            rois = [
                # High Left hand / Raised stone position
                (max(0, px1 - int(pw * 0.35)), max(0, py1 - int(ph * 0.15)),
                 min(w, px1 + int(pw * 0.50)), min(h, py1 + int(ph * 0.65))),
                # High Right hand / Raised stone position
                (max(0, px2 - int(pw * 0.50)), max(0, py1 - int(ph * 0.15)),
                 min(w, px2 + int(pw * 0.35)), min(h, py1 + int(ph * 0.65))),
                # Mid/Low Left hand region
                (max(0, px1 - int(pw * 0.25)), max(0, py1 + int(ph * 0.25)),
                 min(w, px1 + int(pw * 0.45)), min(h, py1 + int(ph * 0.90))),
                # Mid/Low Right hand region
                (max(0, px2 - int(pw * 0.45)), max(0, py1 + int(ph * 0.25)),
                 min(w, px2 + int(pw * 0.25)), min(h, py1 + int(ph * 0.90))),
                # Center chest/shoulder / tool hold region
                (max(0, px1 - int(pw * 0.10)), max(0, py1 - int(ph * 0.05)),
                 min(w, px2 + int(pw * 0.10)), min(h, py1 + int(ph * 0.75))),
                # Footstep payload / dropped stone region
                (max(0, px1 - int(pw * 0.15)), max(0, py2 - int(ph * 0.18)),
                 min(w, px2 + int(pw * 0.15)), min(h, py2 + int(ph * 0.15))),
            ]

            for rx1, ry1, rx2, ry2 in rois:
                if rx2 <= rx1 or ry2 <= ry1:
                    continue

                # ── Strategy A: Edge-based (for normal contrast scenes) ─────────────────
                candidate_mask_edge = None
                roi_gray = gray[ry1:ry2, rx1:rx2]
                if roi_gray.size > 0:
                    # Lower thresholds for foggy: use 10/50 instead of 30/100
                    lo_th, hi_th = (10, 50) if is_adverse_scene else (30, 100)
                    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
                    canny = cv2.Canny(roi_blur, lo_th, hi_th)
                    lap = cv2.convertScaleAbs(cv2.Laplacian(roi_blur, cv2.CV_64F))
                    combined = cv2.bitwise_or(
                        canny, cv2.threshold(lap, 12, 255, cv2.THRESH_BINARY)[1]
                    )
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    candidate_mask_edge = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

                # ── Strategy B: Color-based dark-blob against snow/fog backgrounds ──────
                # Stones are typically dark (low L in Lab) or warm-brown (hue 10-30 in HSV).
                # Against bright snow/fog background they create a strong blob.
                candidate_mask_color = None
                roi_lab = lab_img[ry1:ry2, rx1:rx2]
                roi_hsv = hsv_img[ry1:ry2, rx1:rx2]
                if roi_lab.size > 0:
                    # Dark objects: L channel (Lab) < 130 in bright scene
                    L_channel = roi_lab[:, :, 0]
                    if is_adverse_scene and mean_lum > 100:
                        # Snow background: stones/objects stand out as dark patches
                        # Threshold: pixels darker than 70% of mean scene brightness
                        dark_thresh = min(160, int(mean_lum * 0.70))
                        dark_mask = (L_channel < dark_thresh).astype(np.uint8) * 255
                    else:
                        # General: isolate anything below scene median brightness
                        scene_median_L = float(np.median(lab_img[:, :, 0]))
                        dark_mask = (L_channel < max(100, scene_median_L * 0.75)).astype(np.uint8) * 255

                    # Warm brown/orange tone mask (stone/rock typical hue 5–35)
                    warm_mask = cv2.inRange(roi_hsv, np.array([5, 30, 30]), np.array([35, 255, 220]))

                    # Combine dark + warm masks
                    color_combined = cv2.bitwise_or(dark_mask, warm_mask)

                    # Morphological cleanup
                    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    candidate_mask_color = cv2.morphologyEx(
                        cv2.morphologyEx(color_combined, cv2.MORPH_OPEN, kernel_c),
                        cv2.MORPH_CLOSE, kernel_c
                    )

                # ── Merge both candidate masks ────────────────────────────────────────────
                candidate_mask = None
                if candidate_mask_edge is not None and candidate_mask_color is not None:
                    candidate_mask = cv2.bitwise_or(candidate_mask_edge, candidate_mask_color)
                elif candidate_mask_edge is not None:
                    candidate_mask = candidate_mask_edge
                elif candidate_mask_color is not None:
                    candidate_mask = candidate_mask_color

                if candidate_mask is None:
                    continue

                contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Handheld object size constraint
                    if 30 <= area <= int(pw * ph * 0.28):
                        cx, cy, cw, ch = cv2.boundingRect(cnt)
                        aspect = float(cw) / max(1, float(ch))
                        if 0.25 <= aspect <= 4.0 and cw >= 7 and ch >= 7:
                            ox1 = float(rx1 + cx)
                            oy1 = float(ry1 + cy)
                            ox2 = float(rx1 + cx + cw)
                            oy2 = float(ry1 + cy + ch)

                            # Check for near-duplicates
                            is_dup = any(
                                abs(ox1 - ed.bbox[0]) < 20 and abs(oy1 - ed.bbox[1]) < 20
                                for ed in handheld_dets
                            )

                            if not is_dup:
                                # Classify: vertical cylinder/bottle vs compact stone/tool
                                obj_cls_name = "bottle" if ch >= 1.25 * cw else "stone"
                                obj_cls_id = 39 if obj_cls_name == "bottle" else 88

                                # Crop-level verification with YOLO if available
                                try:
                                    crop_pad = 6
                                    c_y1 = max(0, int(oy1) - crop_pad)
                                    c_y2 = min(h, int(oy2) + crop_pad)
                                    c_x1 = max(0, int(ox1) - crop_pad)
                                    c_x2 = min(w, int(ox2) + crop_pad)
                                    if (c_y2 - c_y1) >= 14 and (c_x2 - c_x1) >= 14:
                                        crop_img = image_np[c_y1:c_y2, c_x1:c_x2]
                                        crop_res = self.model(crop_img, conf=0.08, verbose=False)
                                        for cr in crop_res:
                                            if cr.boxes is not None and len(cr.boxes) > 0:
                                                c_cid = int(cr.boxes.cls[0].cpu().numpy())
                                                c_name = self.model.names.get(c_cid, "").lower()
                                                if c_name in ("bottle", "cup", "cell phone", "knife"):
                                                    obj_cls_name = c_name
                                                    obj_cls_id = c_cid
                                                    break
                                except Exception:
                                    pass

                                handheld_dets.append(
                                    Detection(
                                        class_id=obj_cls_id,
                                        class_name=obj_cls_name,
                                        confidence=0.91,
                                        bbox=[round(ox1, 2), round(oy1, 2),
                                              round(ox2, 2), round(oy2, 2)],
                                        camera_id=camera_id,
                                        timestamp=timestamp,
                                        frame_id=frame_id,
                                        source=DetectionSource.SCENE_ANALYSIS
                                    )
                                )
                                break  # One candidate per ROI

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

        effective_conf = max(0.15, self.confidence_threshold - 0.08) if is_low_contrast else self.confidence_threshold

        # 2. Run YOLO inference
        results = self.model(
            yolo_input,
            conf=min(0.08, effective_conf),
            iou=0.45,
            agnostic_nms=True,
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

                # Lower threshold (0.08) specifically for animals (dog/cat) and held items (bottle) on snow/fog feeds
                min_conf = 0.08 if cls_name in ("dog", "cat", "bottle") else effective_conf
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

        # 2. Apply Sub-Box Containment Suppression
        clean_detections = suppress_nested_subboxes(raw_detections, containment_threshold=0.60)

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
        ground_animal_dets = self._detect_ground_animals(
            image_np,
            clean_detections,
            camera_id=frame_obj.camera_id,
            timestamp=frame_obj.timestamp,
            frame_id=frame_obj.frame_id
        )
        if ground_animal_dets:
            clean_detections.extend(ground_animal_dets)

        # 4. Detect Perimeter Fence & Boundary Structures
        fence_dets = self.fence_detector.detect_fence(
            image_np,
            camera_id=frame_obj.camera_id,
            timestamp=frame_obj.timestamp,
            frame_id=frame_obj.frame_id,
            existing_detections=clean_detections
        )
        if fence_dets:
            clean_detections.extend(fence_dets)

        # 5. Refine Quadruped Animals & Physical Aspect Ratios
        h, w = image_np.shape[:2]
        clean_detections = refine_detection_classes(clean_detections, h, w)

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
        Detects ground-level quadruped animals (cats, dogs) on snow/fog or light backgrounds
        when standard YOLO misses them or produces low-confidence candidates.
        """
        if image_np is None:
            return []

        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np.copy()
        mean_lum = float(np.mean(gray))

        # Focus on ground region (lower 65% of frame)
        ground_y1 = int(h * 0.35)
        ground_roi = gray[ground_y1:, :]
        roi_h, roi_w = ground_roi.shape[:2]

        if roi_h <= 0 or roi_w <= 0:
            return []

        # Isolate dark quadruped blobs against bright snow/ground
        if mean_lum > 75.0:
            dark_thresh = min(175, int(mean_lum * 0.82))
            _, dark_mask = cv2.threshold(ground_roi, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        else:
            _, dark_mask = cv2.threshold(ground_roi, 70, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        animal_dets: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Typical quadruped animal contour area
            if 100 <= area <= int(w * h * 0.12):
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / max(1.0, float(ch))

                # Quadrupeds aspect ratio: frontal, angled, or walking (0.30 to 3.6)
                if 0.30 <= aspect_ratio <= 3.6 and cw >= 12 and ch >= 10:
                    abs_x1 = float(cx)
                    abs_y1 = float(ground_y1 + cy)
                    abs_x2 = float(cx + cw)
                    abs_y2 = float(ground_y1 + cy + ch)

                    # Avoid duplicate if existing detection already covers this exact candidate
                    already_covered = False
                    for det in existing_detections:
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

                    # Check that it doesn't heavily overlap existing person bounding boxes
                    overlap_person = False
                    for det in existing_detections:
                        if det.class_name == "person":
                            px1, py1, px2, py2 = det.bbox
                            ix1 = max(abs_x1, px1)
                            iy1 = max(abs_y1, py1)
                            ix2 = min(abs_x2, px2)
                            iy2 = min(abs_y2, py2)
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            box_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                            if inter / max(1.0, box_area) > 0.40:
                                overlap_person = True
                                break

                    if not overlap_person:
                        # Optional YOLO crop classification
                        detected_label = "dog"
                        detected_conf = 0.90
                        try:
                            pad = 8
                            cy1 = max(0, int(abs_y1) - pad)
                            cy2 = min(h, int(abs_y2) + pad)
                            cx1 = max(0, int(abs_x1) - pad)
                            cx2 = min(w, int(abs_x2) + pad)
                            if (cy2 - cy1) >= 16 and (cx2 - cx1) >= 16:
                                animal_crop = image_np[cy1:cy2, cx1:cx2]
                                c_results = self.model(animal_crop, conf=0.06, verbose=False)
                                for cr in c_results:
                                    if cr.boxes is not None and len(cr.boxes) > 0:
                                        c_id = int(cr.boxes.cls[0].cpu().numpy())
                                        c_n = self.model.names.get(c_id, "").lower()
                                        if c_n in ("dog", "cat", "horse", "sheep", "cow"):
                                            detected_label = c_n if c_n in ("dog", "cat") else "dog"
                                            detected_conf = max(0.90, float(cr.boxes.conf[0].cpu().numpy()))
                                            break
                        except Exception:
                            pass

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
                        if len(animal_dets) >= 2:
                            break

        return animal_dets


def refine_detection_classes(detections: List[Detection], img_height: int, img_width: int) -> List[Detection]:
    """
    Refines class labels for detections based on physical aspect ratios and relative scale.
    Corrects small quadruped animals (dogs/cats) misclassified as 'person' by low-resolution YOLO.
    """
    person_boxes = [d for d in detections if d.class_name == "person"]
    if not person_boxes:
        return detections

    max_person_h = max(d.bbox[3] - d.bbox[1] for d in person_boxes)

    refined: List[Detection] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        aspect_ratio = bh / bw

        if d.class_name == "person":
            # Small ground-level quadruped entity: height < 50% of max human height and aspect ratio < 1.45, or very small
            is_small_ground = (bh < 0.50 * max_person_h) and (y2 > img_height * 0.30)
            is_horizontal_body = (aspect_ratio < 1.45)
            is_very_small = (bh < 0.32 * max_person_h) and (y2 > img_height * 0.38)

            if (is_small_ground and is_horizontal_body) or is_very_small:
                d.class_name = "dog"
                d.class_id = 16

        refined.append(d)

    return refined
