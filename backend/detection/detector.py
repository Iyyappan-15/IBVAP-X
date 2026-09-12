import os
import logging
from typing import List
import cv2
import numpy as np
import torch

from backend.config import settings
from backend.interfaces import Detection, Frame

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    """Raised when YOLO weights file is missing."""
    pass

class ObjectDetector:
    """Wrapper around Ultralytics YOLOv8 for Object Detection."""

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        target_classes: List[str] = None
    ):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.YOLO_CONFIDENCE_THRESHOLD
        self.target_classes = target_classes or settings.detect_classes_list

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
        Runs object detection on a Frame object or numpy array image.
        Returns a list of Detection objects.
        """
        if image_np is None:
            if frame_obj.frame_bytes is not None:
                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                logger.warning(f"[Detector] No image content available for frame {frame_obj.frame_id}")
                return []

        # Run inference
        # iou=0.45  : strict NMS — merges overlapping boxes of the SAME object
        #             (default 0.7 is too permissive; causes 1 car → 2 boxes)
        # agnostic_nms=True : also suppresses cross-class overlaps
        #                     (prevents car box swallowing a person box next to it)
        results = self.model(
            image_np,
            conf=self.confidence_threshold,
            iou=0.45,
            agnostic_nms=True,
            classes=self.target_class_ids if self.target_class_ids else None,
            verbose=False
        )


        detections: List[Detection] = []

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
                detections.append(det)

        return detections
