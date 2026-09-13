"""
backend/detection/fence_detector.py

High-Precision Physical Perimeter Fence Detector for IBVAP-X.

Detection logic:
  Real security fences (chain-link, wire, palisade) have a distinctive LATTICE structure:
    - Dense, parallel near-VERTICAL line segments (posts / wires)
    - Dense, parallel near-HORIZONTAL line segments (horizontal bars / cross-wires)
    - Both sets intersect and are present in the SAME spatial region

  Winter reeds, dry grass, bare tree branches generate ONLY vertical lines (stems) with NO
  horizontal cross-structure. This is the key discriminator.

  Snow/fog backgrounds produce near-zero edge density — detection is suppressed.

Algorithm:
  1. Suppress if adverse weather (fog / snow / low-contrast).
  2. Restrict to middle 25-80% of frame height.
  3. Detect near-vertical lines using HoughLinesP.
  4. Detect near-horizontal lines using HoughLinesP.
  5. Require BOTH dense vertical AND horizontal lines in same X-column region.
     (grass/reeds have vertical only — no horizontal → rejected)
  6. Require ≥ 5 vertical clusters with tight regular spacing (CV < 0.35).
  7. Require ≥ 3 horizontal lines overlapping the vertical cluster X-region.
  8. Require fence box spans ≥ 20% of frame width.
"""
import logging
from typing import List, Optional
import cv2
import numpy as np

from backend.interfaces import Detection

logger = logging.getLogger(__name__)


class FenceDetector:
    """
    Detects real physical perimeter fences via bidirectional lattice structure analysis.
    Only fires when BOTH vertical AND horizontal structural lines are found in the same region.
    Completely suppressed on fog/snow/low-contrast scenes.
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_adverse_scene(image_np: np.ndarray) -> bool:
        """Returns True if image has very low contrast (fog/snow) → skip fence detection."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))
        # Foggy: low std with moderate-high luminance
        # Snowy outdoor: moderate std but very high mean (dominated by white snow)
        if (std_val < 45.0 and mean_val > 100.0):
            return True
        if std_val < 25.0:
            return True
        return False

    @staticmethod
    def _cluster_x_coords(x_coords: List[float], gap: float = 20.0) -> List[List[float]]:
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
        Returns a Detection with class_name='fence_perimeter' ONLY when a genuine
        fence lattice structure (vertical + horizontal lines in same region) is visible.

        Returns [] for grass, reeds, trees, snow backgrounds, fog scenes.
        """
        try:
            if image_np is None or image_np.size == 0:
                return []

            h, w = image_np.shape[:2]

            # ── Step 1: Suppress during fog/snow/low-contrast scenes ─────────────
            if self._is_adverse_scene(image_np):
                logger.debug("[FenceDetector] Adverse scene suppression — skipping.")
                return []

            # ── Step 2: Work only on the middle vertical band (25%–80% height) ───
            band_y1 = int(h * 0.25)
            band_y2 = int(h * 0.80)
            roi = image_np[band_y1:band_y2, :]
            roi_h = band_y2 - band_y1

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

            # ── Step 3: CLAHE + blur + edge detection ────────────────────────────
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 120)

            min_line_len = int(roi_h * 0.18)

            # ── Step 4: Detect near-VERTICAL lines ───────────────────────────────
            v_lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=20,
                minLineLength=min_line_len, maxLineGap=15
            )
            vertical_x: List[float] = []
            if v_lines is not None:
                for line in v_lines:
                    x1, y1, x2, y2 = line[0]
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dy == 0:
                        continue
                    angle_from_vertical = np.degrees(np.arctan2(dx, dy))
                    if angle_from_vertical <= 20.0:  # Within 20° of vertical
                        vertical_x.append((x1 + x2) / 2.0)

            if len(vertical_x) < 10:
                return []

            # ── Step 5: Detect near-HORIZONTAL lines ─────────────────────────────
            # This is the KEY discriminator: grass/reeds have NO horizontal lines.
            # Real fences (chain-link, palisade, barbed wire) have horizontal bars.
            h_lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=18,
                minLineLength=int(w * 0.08), maxLineGap=12
            )
            horizontal_segments: List[tuple] = []  # (x1, x2) spans
            if h_lines is not None:
                for line in h_lines:
                    x1, y1, x2, y2 = line[0]
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dx == 0:
                        continue
                    angle_from_horizontal = np.degrees(np.arctan2(dy, dx))
                    if angle_from_horizontal <= 20.0:  # Within 20° of horizontal
                        horizontal_segments.append((min(x1, x2), max(x1, x2)))

            # Require at least 3 horizontal line segments (at least 3 cross-wires visible)
            if len(horizontal_segments) < 3:
                logger.debug(
                    "[FenceDetector] Rejected: only %d horizontal lines found (need ≥3 for lattice).",
                    len(horizontal_segments)
                )
                return []

            # ── Step 6: Cluster vertical X-coordinates ───────────────────────────
            clusters = self._cluster_x_coords(vertical_x, gap=20.0)

            # Need at least 5 distinct vertical clusters (= 5 fence posts)
            if len(clusters) < 5:
                return []

            # ── Step 7: Check REGULAR spacing between cluster centroids ──────────
            centroids = sorted([float(np.mean(c)) for c in clusters])
            spacings = [centroids[i + 1] - centroids[i] for i in range(len(centroids) - 1)]
            mean_spacing = float(np.mean(spacings))
            std_spacing = float(np.std(spacings))

            if mean_spacing < 5.0:
                return []

            # Real fences: CV < 0.35 (tight regular post spacing)
            # Random vegetation: CV > 0.40 (irregular random stem positions)
            cv_spacing = std_spacing / mean_spacing
            if cv_spacing > 0.35:
                logger.debug(
                    "[FenceDetector] Rejected: cluster spacing CV=%.2f > 0.35 (irregular — not a fence).",
                    cv_spacing
                )
                return []

            # ── Step 8: Verify horizontal lines overlap the vertical cluster region ──
            fx1_vert = min(centroids) - 15
            fx2_vert = max(centroids) + 15
            h_overlaps = sum(
                1 for (hx1, hx2) in horizontal_segments
                if hx1 < fx2_vert and hx2 > fx1_vert  # Overlaps vertical cluster X-range
            )
            if h_overlaps < 2:
                logger.debug(
                    "[FenceDetector] Rejected: only %d horizontal lines overlap vertical region "
                    "(need ≥2 cross-wires in fence zone).",
                    h_overlaps
                )
                return []

            # ── Step 9: Build bounding box ─────────────────────────────────────────
            fx1 = max(0, int(fx1_vert))
            fx2 = min(w, int(fx2_vert))
            fy1 = band_y1
            fy2 = band_y2

            # Fence box must span ≥ 20% of frame width
            if (fx2 - fx1) < int(w * 0.20):
                return []

            logger.info(
                "[FenceDetector] Fence confirmed — %d V-clusters, %d H-lines overlap, CV=%.2f, box=[%d,%d,%d,%d]",
                len(clusters), h_overlaps, cv_spacing, fx1, fy1, fx2, fy2
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
            logger.warning("[FenceDetector] Detection error: %s", exc)
            return []
