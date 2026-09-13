import os
import logging
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import torch

from backend.config import settings
from backend.interfaces import Detection, Track, DetectionSource

logger = logging.getLogger(__name__)

class OpenVocabModelNotFoundError(Exception):
    """Raised when YOLO-World model checkpoint is missing and auto-download is disabled."""
    pass

class OpenVocabEngine:
    """
    Dual-Path Open-Vocabulary Semantic Engine using Ultralytics YOLO-World v2.
    Provides (1) Track Crop Refinement and (2) Full-Frame Keyframe Discovery.
    """

    PROMPT_PRESETS: Dict[str, List[str]] = {
        "perimeter": [
            "person", "dog", "vehicle", "fence", "chain link fence", "gate",
            "rock", "stone", "backpack", "bicycle", "motorcycle", "pole", "border marker"
        ],
        "general": [
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "dog", "cat", "backpack", "bag"
        ]
    }

    def __init__(
        self,
        model_path: str = None,
        auto_download: bool = None,
        confidence_threshold: float = 0.25
    ):
        self.model_path = model_path or getattr(settings, "OPEN_VOCAB_MODEL_PATH", "data/models/yolov8s-worldv2.pt")
        self.auto_download = auto_download if auto_download is not None else getattr(settings, "OPEN_VOCAB_AUTO_DOWNLOAD", False)
        self.confidence_threshold = confidence_threshold
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_available = False
        self.active_prompts: List[str] = []

        self._initialize_model()

    def _initialize_model(self):
        """Initializes YOLO-World checkpoint if present or auto-download allowed."""
        if not os.path.exists(self.model_path):
            if not self.auto_download:
                logger.warning(
                    f"[OpenVocabEngine] Weights file missing at '{self.model_path}' and auto-download is False."
                )
                self.is_available = False
                return
            else:
                logger.info(f"[OpenVocabEngine] Auto-downloading YOLO-World model to '{self.model_path}'...")
                os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)

        try:
            from ultralytics import YOLOWorld
            model_file = self.model_path if os.path.exists(self.model_path) else "yolov8s-worldv2.pt"
            self.model = YOLOWorld(model_file)
            self.model.to(self.device)
            self.is_available = True
            
            # Save downloaded file to target path if downloaded in current working directory
            if not os.path.exists(self.model_path) and os.path.exists("yolov8s-worldv2.pt"):
                import shutil
                shutil.copy("yolov8s-worldv2.pt", self.model_path)

            # Set default preset classes
            preset_default = getattr(settings, "PROMPT_PRESET_DEFAULT", "perimeter")
            self.set_classes(self.PROMPT_PRESETS.get(preset_default, self.PROMPT_PRESETS["perimeter"]))
            logger.info(f"[OpenVocabEngine] Model successfully initialized on {self.device}")
        except Exception as e:
            logger.error(f"[OpenVocabEngine] Failed to initialize YOLO-World model: {e}")
            self.is_available = False

    def set_classes(self, prompts: List[str]):
        """Dynamically configures active target class vocabulary for open-vocab discovery."""
        if not self.is_available or self.model is None:
            self.active_prompts = prompts
            return

        cleaned_prompts = [p.strip().lower() for p in prompts if p and p.strip()]
        if not cleaned_prompts:
            cleaned_prompts = self.PROMPT_PRESETS["perimeter"]

        self.active_prompts = cleaned_prompts
        try:
            self.model.set_classes(self.active_prompts)
            logger.info(f"[OpenVocabEngine] Active prompts set to: {self.active_prompts}")
        except Exception as e:
            logger.error(f"[OpenVocabEngine] Error calling set_classes: {e}")

    def discover_full_frame(
        self,
        image_np: np.ndarray,
        prompts: Optional[List[str]] = None,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 0
    ) -> List[Detection]:
        """
        Runs full-frame open-vocabulary discovery on keyframes.
        Returns candidate detections tagged with DetectionSource.OPEN_VOCAB_DISCOVERY.
        """
        if not self.is_available or self.model is None or image_np is None:
            return []

        if prompts and prompts != self.active_prompts:
            self.set_classes(prompts)

        h, w = image_np.shape[:2]
        effective_conf = self.confidence_threshold

        try:
            results = self.model(
                image_np,
                conf=effective_conf,
                iou=0.45,
                agnostic_nms=True,
                verbose=False
            )
        except Exception as e:
            logger.error(f"[OpenVocabEngine] Full-frame discovery error: {e}")
            return []

        discoveries: List[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Retrieve label name from active prompts or model names
                if hasattr(self.model, "names") and cls_id in self.model.names:
                    cls_name = self.model.names[cls_id]
                elif cls_id < len(self.active_prompts):
                    cls_name = self.active_prompts[cls_id]
                else:
                    cls_name = f"object_{cls_id}"

                cls_name = cls_name.lower().strip()

                # Experimental unclassified flagging
                if getattr(settings, "ENABLE_UNCLASSIFIED_FLAGGING", False) and conf < 0.20:
                    cls_name = "UNCLASSIFIED — REVIEW"

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[round(v, 2) for v in xyxy],
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    source=DetectionSource.OPEN_VOCAB_DISCOVERY
                )
                discoveries.append(det)

        return discoveries

    def refine_track_crops(
        self,
        image_np: np.ndarray,
        active_tracks: List[Track],
        prompts: Optional[List[str]] = None,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 0
    ) -> List[Detection]:
        """
        Runs open-vocabulary inference on cropped bounding boxes of active tracks
        to refine semantic labels (e.g. correcting dog vs person).
        """
        if not self.is_available or self.model is None or image_np is None or not active_tracks:
            return []

        if prompts and prompts != self.active_prompts:
            self.set_classes(prompts)

        h, w = image_np.shape[:2]
        refined_detections: List[Detection] = []

        for track in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            # Add small padding around crop
            pad_w = int((x2 - x1) * 0.10)
            pad_h = int((y2 - y1) * 0.10)

            cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)

            crop = image_np[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            try:
                results = self.model(
                    crop,
                    conf=0.20,
                    iou=0.45,
                    verbose=False
                )
            except Exception:
                continue

            best_conf = 0.0
            best_label = None

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    if cls_id < len(self.active_prompts):
                        label = self.active_prompts[cls_id]
                    elif hasattr(self.model, "names") and cls_id in self.model.names:
                        label = self.model.names[cls_id]
                    else:
                        label = f"object_{cls_id}"

                    if conf > best_conf:
                        best_conf = conf
                        best_label = label.lower().strip()

            if best_label and best_conf >= 0.25:
                det = Detection(
                    class_id=track.track_id,
                    class_name=best_label,
                    confidence=round(best_conf, 4),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    camera_id=camera_id,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    source=DetectionSource.OPEN_VOCAB_REFINEMENT
                )
                refined_detections.append(det)

        return refined_detections

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata status of OpenVocabEngine for diagnostics."""
        return {
            "is_available": self.is_available,
            "model_path": self.model_path,
            "device": self.device,
            "auto_download": self.auto_download,
            "active_prompts": self.active_prompts,
            "confidence_threshold": self.confidence_threshold
        }
