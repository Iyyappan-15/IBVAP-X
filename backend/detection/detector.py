import os
import logging
from typing import List
import cv2
import numpy as np
import torch

from backend.config import settings
from backend.interfaces import Detection, Frame
from backend.detection.fence_detector import FenceDetector

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    """Raised when YOLO weights file is missing."""
    pass


def suppress_nested_subboxes(detections: List[Detection], containment_threshold: float = 0.60) -> List[Detection]:
    """
    Suppresses nested / containment duplicate detections (e.g. YOLO detecting a car window/door
    as an extra car inside a larger car box).
    If Box B is >= containment_threshold (60%) contained inside Box A of the same class/vehicle category,
    the smaller Box B is suppressed.
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

            if containment >= containment_threshold and area_a > area_b:
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

        # Expanded surveillance classes (including handheld objects, luggage, tools)
        self.surveillance_classes = {
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "backpack", "handbag", "suitcase", "sports ball", "bottle",
            "knife", "baseball bat", "cell phone", "umbrella", "scissors"
        }
        if target_classes:
            self.target_classes = target_classes
        else:
            self.target_classes = list(self.surveillance_classes.union(set(settings.detect_classes_list)))

        # Map class names to class IDs for configured target classes
        self.target_class_ids = []
        if hasattr(self.model, "names"):
            for cid, cname in self.model.names.items():
                if cname.lower() in [tc.lower() for tc in self.target_classes]:
                    self.target_class_ids.append(cid)

    def _detect_handheld_objects(
        self,
        image_np: np.ndarray,
        person_detections: List[Detection],
        camera_id: str,
        timestamp: float,
        frame_id: int
    ) -> List[Detection]:
        """
        Detects compact handheld objects (stones, tools, thrown items) in or near the hand regions of detected persons.
        """
        if image_np is None or not person_detections:
            return []

        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        handheld_dets: List[Detection] = []

        for p_det in person_detections:
            px1, py1, px2, py2 = [int(v) for v in p_det.bbox]
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)

            # Hand regions: left and right lateral torso sectors
            hand_rois = [
                # Left hand region
                (max(0, px1 - int(pw * 0.30)), max(0, py1 + int(ph * 0.35)), min(w, px1 + int(pw * 0.40)), min(h, py1 + int(ph * 0.85))),
                # Right hand region
                (max(0, px2 - int(pw * 0.40)), max(0, py1 + int(ph * 0.35)), min(w, px2 + int(pw * 0.30)), min(h, py1 + int(ph * 0.85))),
            ]

            for rx1, ry1, rx2, ry2 in hand_rois:
                if rx2 <= rx1 or ry2 <= ry1:
                    continue

                roi_gray = gray[ry1:ry2, rx1:rx2]
                if roi_gray.size == 0:
                    continue

                # Detect salient compact objects using thresholding & contour analysis
                roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
                # Adaptive gradient / Laplacian for high-contrast stone/object texture
                lap = cv2.Laplacian(roi_blur, cv2.CV_64F)
                lap_abs = cv2.convertScaleAbs(lap)
                _, thresh = cv2.threshold(lap_abs, 20, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Stone/handheld object size limits
                    if 120 <= area <= int(pw * ph * 0.18):
                        cx, cy, cw, ch = cv2.boundingRect(cnt)
                        aspect = float(cw) / max(1, float(ch))
                        if 0.4 <= aspect <= 2.5 and cw >= 14 and ch >= 14:
                            ox1 = float(rx1 + cx)
                            oy1 = float(ry1 + cy)
                            ox2 = float(rx1 + cx + cw)
                            oy2 = float(ry1 + cy + ch)

                            # Ensure it's not a duplicate
                            is_dup = False
                            for ed in handheld_dets:
                                ex1, ey1, ex2, ey2 = ed.bbox
                                if abs(ox1 - ex1) < 20 and abs(oy1 - ey1) < 20:
                                    is_dup = True
                                    break

                            if not is_dup:
                                handheld_dets.append(
                                    Detection(
                                        class_id=88,
                                        class_name="stone",
                                        confidence=0.88,
                                        bbox=[round(ox1, 2), round(oy1, 2), round(ox2, 2), round(oy2, 2)],
                                        camera_id=camera_id,
                                        timestamp=timestamp,
                                        frame_id=frame_id
                                    )
                                )
                                break  # One primary handheld object per hand

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

        # 1. Adaptive contrast enhancement for night / low-light CCTV
        yolo_input = image_np
        try:
            gray_check = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_check) < 70.0:
                lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                yolo_input = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            yolo_input = image_np

        # 2. Run YOLO inference
        results = self.model(
            yolo_input,
            conf=self.confidence_threshold,
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
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                if conf < self.confidence_threshold:
                    continue

                # Map sports ball / bottle / phone near hand to stone/object if relevant
                if cls_name.lower() in ["sports ball", "frisbee"]:
                    cls_name = "stone"

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[round(v, 2) for v in xyxy],
                    camera_id=frame_obj.camera_id,
                    timestamp=frame_obj.timestamp,
                    frame_id=frame_obj.frame_id
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

        return clean_detections
