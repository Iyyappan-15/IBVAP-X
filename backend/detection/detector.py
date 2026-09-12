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

        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(
                f"YOLO model weights file not found at '{self.model_path}'. "
                f"Run 'python scripts/download_model.py' to download weights."
            )

        # Detect GPU vs CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Detector] Initializing YOLO model '{self.model_path}' on device: {self.device}")

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Map class names to class IDs for configured target classes
        self.target_class_ids = []
        if hasattr(self.model, "names"):
            for cid, cname in self.model.names.items():
                if cname.lower() in [tc.lower() for tc in self.target_classes]:
                    self.target_class_ids.append(cid)

    def detect(self, frame_obj: Frame, image_np: np.ndarray = None) -> List[Detection]:
        """
        Runs object detection + fence detection on a Frame or image.
        Returns clean Detection objects without nested duplicate boxes.
        """
        if image_np is None:
            if frame_obj.frame_bytes is not None:
                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                logger.warning(f"[Detector] No image content available for frame {frame_obj.frame_id}")
                return []

        # 1. Run YOLO inference
        results = self.model(
            image_np,
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

        # 3. Detect Perimeter Fence & Boundary Structures
        fence_dets = self.fence_detector.detect_fence(
            image_np,
            camera_id=frame_obj.camera_id,
            timestamp=frame_obj.timestamp,
            frame_id=frame_obj.frame_id
        )
        if fence_dets:
            clean_detections.extend(fence_dets)

        return clean_detections
