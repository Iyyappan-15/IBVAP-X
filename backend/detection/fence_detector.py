import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from backend.interfaces import Detection, DetectionSource

logger = logging.getLogger(__name__)

class FenceDetector:
    """
    Detects real physical perimeter chain-link fences and boundary structures via
    bidirectional lattice structure analysis and diamond wire-mesh pattern verification.
    """

    def __init__(self, confidence: float = 0.92):
        self.confidence = confidence

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
        fence lattice or chain-link mesh structure is visible in the frame.
        """
        try:
            if image_np is None or image_np.size == 0:
                return []

            h, w = image_np.shape[:2]

            # Work on middle vertical band (15%–85% height) where fences reside
            band_y1 = int(h * 0.15)
            band_y2 = int(h * 0.85)
            roi = image_np[band_y1:band_y2, :]
            roi_h = band_y2 - band_y1

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

            # Adaptive contrast enhancement (CLAHE) for snowy/foggy CCTV feeds
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # ── 1. Detect Near-Vertical Posts & Poles ───────────────────────────
            min_line_len = int(roi_h * 0.15)
            v_lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=20,
                minLineLength=min_line_len, maxLineGap=20
            )
            vertical_x: List[float] = []
            if v_lines is not None:
                for line in v_lines:
                    x1, y1, x2, y2 = line.ravel()
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dy == 0:
                        continue
                    angle_vert = np.degrees(np.arctan2(dx, dy))
                    if angle_vert <= 22.0:  # Within 22° of vertical
                        vertical_x.append((x1 + x2) / 2.0)

            # ── 2. Detect Near-Horizontal Cross-Wires / Top Rails ───────────────
            h_lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=18,
                minLineLength=int(w * 0.06), maxLineGap=15
            )
            horizontal_segments: List[Tuple[float, float]] = []
            if h_lines is not None:
                for line in h_lines:
                    x1, y1, x2, y2 = line.ravel()
                    dx, dy = abs(x2 - x1), abs(y2 - y1)
                    if dx == 0:
                        continue
                    angle_horiz = np.degrees(np.arctan2(dy, dx))
                    if angle_horiz <= 22.0:  # Within 22° of horizontal
                        horizontal_segments.append((min(x1, x2), max(x1, x2)))

            # ── 3. Detect Bidirectional Diamond Mesh Lines (+45° / -45°) ──────────
            d_lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180, threshold=15,
                minLineLength=int(roi_h * 0.08), maxLineGap=10
            )
            diag_pos = 0
            diag_neg = 0
            if d_lines is not None:
                for line in d_lines:
                    x1, y1, x2, y2 = line.ravel()
                    dx, dy = (x2 - x1), (y2 - y1)
                    if dx == 0:
                        continue
                    slope = dy / float(dx)
                    angle = np.degrees(np.arctan(abs(slope)))
                    if 25.0 <= angle <= 65.0:
                        if slope > 0:
                            diag_pos += 1
                        else:
                            diag_neg += 1
            diag_count = diag_pos + diag_neg

            # Genuine perimeter chain-link fence requires bidirectional diamond cross-hatching or multiple rails
            has_diamond_mesh = (diag_pos >= 12 and diag_neg >= 12 and diag_count >= 28)
            has_rails = len(horizontal_segments) >= 3

            if len(vertical_x) < 8 or (not has_diamond_mesh and not has_rails):
                return []

            # Group vertical coordinates into post clusters
            clusters = self._cluster_x_coords(vertical_x, gap=25.0)
            clusters = [c for c in clusters if len(c) >= 2]

            if len(clusters) < 3:
                return []

            # Calculate inter-post spacing regularity (Coefficient of Variation)
            centroids = [float(np.mean(c)) for c in clusters]
            spacings = [centroids[i+1] - centroids[i] for i in range(len(centroids)-1)]
            if not spacings:
                return []

            mean_spacing = float(np.mean(spacings))
            std_spacing = float(np.std(spacings))

            if mean_spacing < 8.0:
                return []

            cv_spacing = std_spacing / mean_spacing
            if cv_spacing > 0.40:  # Require regular spacing (reject random foliage/trees)
                return []

            # Build bounding box for perimeter fence
            fx1 = max(0, int(min(centroids) - 10))
            fx2 = min(w, int(max(centroids) + 10))
            fy1 = band_y1
            fy2 = band_y2

            # Fence must span at least 15% of frame width
            if (fx2 - fx1) < int(w * 0.15):
                return []

            # Require minimum edge mesh density and cross-hatching to reject open snow/trees/horizons
            candidate_roi_edges = edges[:, fx1:fx2]
            edge_density = float(np.count_nonzero(candidate_roi_edges)) / float(max(1, candidate_roi_edges.size))
            if edge_density < 0.045 or (not has_diamond_mesh and not has_rails):
                logger.debug(f"[FenceDetector] Rejected candidate: edge_density={edge_density:.4f}, diags={diag_count} (pos={diag_pos}, neg={diag_neg})")
                return []

            logger.info(f"[FenceDetector] Fence confirmed: [{fx1}, {fy1}, {fx2}, {fy2}] (Clusters: {len(clusters)}, Diags: {diag_count} [pos:{diag_pos}, neg:{diag_neg}], EdgeDensity: {edge_density:.4f})")

            return [Detection(
                class_id=91,
                class_name="fence_perimeter",
                confidence=round(self.confidence, 4),
                bbox=[float(fx1), float(fy1), float(fx2), float(fy2)],
                camera_id=camera_id,
                timestamp=timestamp,
                frame_id=frame_id,
                source=DetectionSource.INFRASTRUCTURE_ANALYSIS
            )]

        except Exception as exc:
            logger.warning(f"[FenceDetector] Detection error: {exc}")
            return []
