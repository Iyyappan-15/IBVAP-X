import os
import stat
import time
import json
import uuid
import logging
import cv2
import numpy as np

from backend.config import settings
from backend.interfaces import EvidenceRecord, AlertOutput
from backend.evidence.hashing import EvidenceHasher
from backend.evidence.audit import AuditLogger

logger = logging.getLogger(__name__)

class EvidenceCapturer:
    """Auto-captures MP4 clip, snapshot JPEG, JSON metadata, computes SHA-256, and seals evidence files."""

    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or settings.EVIDENCE_STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)
        self.audit_logger = AuditLogger()

    def capture_evidence(
        self,
        alert: AlertOutput,
        pre_event_frames: list,
        current_frame_img: np.ndarray,
        fps: float = 10.0
    ) -> EvidenceRecord:
        """
        Captures MP4 clip, JPEG snapshot, JSON metadata, computes SHA-256 hash, and marks read-only sealed.
        """
        evidence_id = f"EVD-{uuid.uuid4().hex[:8].upper()}"
        base_name = f"event_{alert.alert_id}_{evidence_id}"

        snapshot_path = os.path.join(self.storage_dir, f"{base_name}.jpg")
        video_path = os.path.join(self.storage_dir, f"{base_name}.mp4")
        metadata_path = os.path.join(self.storage_dir, f"{base_name}.json")

        # 1. Save Peak Snapshot JPEG
        if current_frame_img is not None and current_frame_img.size > 0:
            cv2.imwrite(snapshot_path, current_frame_img)
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(snapshot_path, blank)

        # 2. Write MP4 Video Clip from pre-event ring buffer
        h, w = (480, 640)
        if current_frame_img is not None and current_frame_img.size > 0:
            h, w = current_frame_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, max(1.0, fps), (w, h))

        # Write pre-event buffered frames
        for _, frame_img in pre_event_frames:
            if frame_img is not None and frame_img.size > 0:
                resized = cv2.resize(frame_img, (w, h))
                out.write(resized)

        # Write current frame 10 times to extend clip
        if current_frame_img is not None and current_frame_img.size > 0:
            for _ in range(10):
                out.write(current_frame_img)

        out.release()

        # 3. Write Metadata JSON
        metadata_content = {
            "evidence_id": evidence_id,
            "alert_id": alert.alert_id,
            "camera_id": alert.camera_id,
            "track_id": alert.track_id,
            "class_name": alert.class_name,
            "timestamp": alert.timestamp,
            "event_priority": alert.event_priority.value,
            "camera_reliability": alert.camera_reliability.value,
            "actionability": alert.actionability.value,
            "why_reasons": alert.why_reasons
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=2)

        # 4. Compute SHA-256 Hash of Snapshot JPEG
        sha256_hash = EvidenceHasher.compute_sha256(snapshot_path)

        # 5. Apply read-only permission flags (Prototype File Locking)
        try:
            os.chmod(snapshot_path, stat.S_IREAD)
            os.chmod(video_path, stat.S_IREAD)
            os.chmod(metadata_path, stat.S_IREAD)
        except Exception as e:
            logger.warning(f"[EvidenceCapturer] Failed to set read-only chmod flags ({e})")

        # 6. Record Audit Log Entry
        self.audit_logger.log_event(
            event_type="EVIDENCE_CAPTURED_AND_SEALED",
            actor="SYSTEM_AUTO_CAPTURE",
            target_id=evidence_id,
            details={"sha256_hash": sha256_hash, "alert_id": alert.alert_id}
        )

        return EvidenceRecord(
            evidence_id=evidence_id,
            alert_id=alert.alert_id,
            event_id=f"EVT-{alert.alert_id}",
            camera_id=alert.camera_id,
            timestamp=alert.timestamp,
            video_filename=os.path.basename(video_path),
            snapshot_filename=os.path.basename(snapshot_path),
            metadata_filename=os.path.basename(metadata_path),
            sha256_hash=sha256_hash,
            is_sealed=True
        )
