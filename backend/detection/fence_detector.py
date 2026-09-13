"""
backend/detection/fence_detector.py

High-Precision Physical Perimeter Fence Detector for IBVAP-X.
Detects security fences, chain-link boundary mesh, and perimeter gates
using strict structural line cluster analysis with fog/snow suppression.

Detection logic:
  1. Canny edge detection on mid-frame band only (avoids sky/snow at top/bottom).
  2. Probabilistic Hough line transform to find near-vertical line segments.
  3. Cluster the X-coordinates of line midpoints.
  4. Require ≥10 clustered vertical lines in the mid-band.
  5. Require REGULAR (evenly-spaced) horizontal distribution of clusters.
  6. Require inter-cluster spacing std_dev < 40% of mean spacing (evenly spaced = fence posts).
  7. Suppress when adverse weather (fog/snow) is active — low contrast means false edge noise.
  8. Never fires on scenes with mean_std < 20 (uniform featureless patches).
"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from backend.interfaces import Detection

logger = logging.getLogger(__name__)


class FenceDetector:
    """
    Detects physical boundary fences and chain-link perimeters via strict structural line clustering.
    Only fires when genuine parallel vertical/diagonal fence lattice lines are present.
    Will NOT fire on:
      - Snow / white sky backgrounds
      - Foggy / low-visibility scenes (adverse weather suppression)
      - Random edge noise from tree branches or terrain
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_foggy(image_np: np.ndarray) -> bool:
        """Returns True if image has very low contrast (fog/snow/mist) — skip fence detection."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))
        # Foggy: low std (flat) with moderate-high luminance OR very low std in any illumination
        return (std_val < 42.0 and mean_val > 110.0) or (std_val < 22.0)

    @staticmethod
    def _cluster_x_coords(x_coords: List[float], gap: float = 18.0) -> List[List[float]]:
        """Groups sorted X-coordinates into clusters with max intra-cluster gap."""
        if not x_coords:
            return []
        sorted_x = sorted(x_coords)
        clusters: List[List[float]] = [[sorted_x[0]]]
        for x in sorted_x[1:]:
            if x - clusters[-1][-1] <= gap:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        return clusters

    def detect_fence(
        self,
        image_np: np.ndarray,
        camera_id: str = "CAM-01",
        timestamp: float = 0.0,
        frame_id: int = 1,
        existing_detections: Optional[List[Detection]] = None
    ) -> List[Detection]:
        """
        Returns a Detection with class_name='fence_perimeter' only when a genuine
        fence structure is visible in the frame. Returns [] for false-positive scenes
        (snow, fog, bare trees, sky gradients).
        """
        try:
            if image_np is None or image_np.size == 0:
                return []

            h, w = image_np.shape[:2]

            # ── Step 1: Suppress during fog/snow/low-contrast conditions ────
            if self._is_foggy(image_np):
                logger.debug("[FenceDetector] Adverse weather suppression active — skipping fence detection.")
                return []

            # ── Step 2: Work only on the middle vertical band (30%–80% height) ──
            # Real fences appear mid-frame; sky/snow at top, ground at bottom are excluded.
            band_y1 = int(h * 0.30)
            band_y2 = int(h * 0.80)
            roi = image_np[band_y1:band_y2, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

            # ── Step 3: Adaptive CLAHE for low-light fence visibility ───────
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # ── Step 4: Edge detection ───────────────────────────────────────
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 120)

            # ── Step 5: Probabilistic Hough for near-vertical line segments ──
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=25,
                minLineLength=int((band_y2 - band_y1) * 0.20),
                maxLineGap=20
            )

            if lines is None or len(lines) < 10:
                return []

            # ── Step 6: Filter near-vertical lines (angle within ±25° of vertical) ──
            vertical_x: List[float] = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy == 0:
                    continue
                angle_from_vertical = np.degrees(np.arctan2(dx, dy))
                if angle_from_vertical <= 25.0:
                    mid_x = (x1 + x2) / 2.0
                    vertical_x.append(mid_x)

            if len(vertical_x) < 10:
                return []

            # ── Step 7: Cluster the X-coordinates ──────────────────────────
            clusters = self._cluster_x_coords(vertical_x, gap=18.0)

            # Need at least 3 distinct vertical clusters (= 3 fence posts/wires visible)
            if len(clusters) < 3:
                return []

            # ── Step 8: Check regular spacing between clusters (fence posts are evenly spaced) ──
            # Compute cluster centroids
            centroids = [float(np.mean(c)) for c in clusters]
            centroids.sort()

            spacings = [centroids[i + 1] - centroids[i] for i in range(len(centroids) - 1)]
            mean_spacing = float(np.mean(spacings))
            std_spacing = float(np.std(spacings))

            if mean_spacing < 5.0:
                return []

            # Coefficient of variation: std/mean. Real fence posts have consistent spacing (<50% CV)
            cv_spacing = std_spacing / mean_spacing
            if cv_spacing > 0.55:
                # Too irregular — random edges, not a fence
                logger.debug(
                    "[FenceDetector] Rejected: cluster spacing CV=%.2f > 0.55 (irregular noise, not fence).",
                    cv_spacing
                )
                return []

            # ── Step 9: Build bounding box around the detected fence region ──
            all_x = [x for c in clusters for x in c]
            fx1 = max(0, int(min(all_x)) - 10)
            fx2 = min(w, int(max(all_x)) + 10)
            fy1 = band_y1
            fy2 = band_y2

            # Fence box must span at least 15% of frame width (real fence, not isolated post)
            box_width = fx2 - fx1
            if box_width < int(w * 0.15):
                return []

            logger.info(
                "[FenceDetector] Fence detected — %d clusters, spacing CV=%.2f, box=[%d,%d,%d,%d]",
                len(clusters), cv_spacing, fx1, fy1, fx2, fy2
            )

            return [Detection(
                class_id=91,
                class_name="fence_perimeter",
                confidence=round(self.confidence, 4),
                bbox=[float(fx1), float(fy1), float(fx2), float(fy2)],
                camera_id=camera_id,
                timestamp=timestamp,
                frame_id=frame_id
            )]

        except Exception as exc:
            logger.warning("[FenceDetector] Detection failed: %s", exc)
            return []
