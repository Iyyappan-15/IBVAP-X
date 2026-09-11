import os
import pickle
import logging
from typing import Optional, List
import numpy as np
from sklearn.ensemble import IsolationForest

from backend.config import settings
from backend.interfaces import Track, AnomalyResult, AnomalyStatus
from backend.anomaly.features import TrajectoryFeatureExtractor

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Unsupervised trajectory anomaly detector wrapping IsolationForest.
    Maps raw IsolationForest output to normalized 0.0-1.0 anomaly index.
    Execution is completely isolated — exceptions are caught and never block the main pipeline.
    """

    def __init__(self, model_path: str = None, contamination: float = None):
        self.model_path = model_path or settings.ANOMALY_MODEL_PATH
        self.contamination = contamination if contamination is not None else settings.ANOMALY_CONTAMINATION
        self.threshold = settings.ANOMALY_SCORE_THRESHOLD
        self.feature_extractor = TrajectoryFeatureExtractor()
        self.model: Optional[IsolationForest] = None

        self._load_or_create_model()

    def _load_or_create_model(self):
        """Loads pre-trained model file if present; otherwise creates fresh model."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"[AnomalyDetector] Loaded pre-trained anomaly model from '{self.model_path}'")
                return
            except Exception as e:
                logger.warning(f"[AnomalyDetector] Failed to load model file '{self.model_path}' ({e}). Creating baseline model.")

        # Initialize baseline model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=50
        )
        # Fit baseline on synthetic normal feature distributions
        dummy_normal_data = np.random.normal(loc=[5.0, 10.0, 0.1, 100.0, 1.0, 0.8, 1.0, 20.0], scale=1.0, size=(100, 8))
        self.model.fit(dummy_normal_data)

    def evaluate_track(self, camera_id: str, track: Track) -> Optional[AnomalyResult]:
        """
        Extracts trajectory features and computes normalized anomaly score.
        Returns AnomalyResult or None if trajectory is too short.
        Isolated: Exceptions log warnings and return None without breaking pipeline execution.
        """
        try:
            feats = self.feature_extractor.extract_features(track)
            if feats is None:
                return None

            feats_2d = feats.reshape(1, -1)
            
            # Scikit-learn score_samples: raw decision score (lower = more anomalous)
            raw_score = float(self.model.score_samples(feats_2d)[0])

            # Refinement 7: Explicit score normalization to 0.0 - 1.0 application anomaly scale
            # Raw score range is roughly -0.5 (very anomalous) to +0.5 (very normal)
            # Normalized score: 0.0 = completely normal, 1.0 = highly anomalous
            norm_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))
            is_anomalous = (norm_score >= self.threshold)
            status = AnomalyStatus.ANOMALOUS if is_anomalous else AnomalyStatus.NORMAL

            explanation = (
                f"Trajectory Anomaly Score: {norm_score:.2f} ({status.value}). "
                f"⚠️ Prototype anomaly model — trained on synthetic baseline trajectories."
            )

            return AnomalyResult(
                track_id=track.track_id,
                camera_id=camera_id,
                timestamp=track.last_seen,
                raw_score=round(raw_score, 4),
                normalized_anomaly_score=round(norm_score, 4),
                status=status,
                is_anomalous=is_anomalous,
                explanation=explanation
            )

        except Exception as e:
            logger.warning(f"[AnomalyDetector] Anomaly evaluation failed for track #{track.track_id} ({e}). Pipeline continuing.")
            return None

    def calibrate_normal_behaviour(self, feature_list: List[np.ndarray]) -> bool:
        """
        Retrains IsolationForest model on current session's normal tracks (Calibration Workflow).
        """
        try:
            if not feature_list or len(feature_list) < 5:
                logger.warning("[AnomalyDetector] Calibration requires at least 5 feature vectors.")
                return False

            X = np.array(feature_list)
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X)

            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            logger.info(f"[AnomalyDetector] Calibrated and saved model with {len(X)} trajectory vectors to '{self.model_path}'")
            return True
        except Exception as e:
            logger.error(f"[AnomalyDetector] Calibration failed ({e})")
            return False
