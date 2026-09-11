#!/usr/bin/env python3
"""
IBVAP-X Anomaly Model Training Script
Generates synthetic normal baseline trajectory features and saves pre-trained IsolationForest model.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest

def train_baseline_model(output_path: str = "data/anomaly_models/ibvapx_anomaly_model.pkl"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Generating synthetic normal baseline trajectories...")

    # Generate 200 synthetic normal trajectory feature vectors
    # Features: [avg_speed, max_speed, stop_ratio, total_distance, direction_changes, curvature, zone_transitions, duration_sec]
    np.random.seed(42)
    normal_speeds = np.random.normal(loc=3.0, scale=0.5, size=(200, 1))
    max_speeds = normal_speeds + np.random.uniform(0.5, 2.0, size=(200, 1))
    stop_ratios = np.random.uniform(0.0, 0.2, size=(200, 1))
    distances = normal_speeds * 30.0 + np.random.normal(0, 5, size=(200, 1))
    dir_changes = np.random.randint(0, 3, size=(200, 1)).astype(float)
    curvatures = np.random.uniform(0.8, 1.0, size=(200, 1))
    zone_trans = np.random.randint(0, 2, size=(200, 1)).astype(float)
    durations = np.random.uniform(10.0, 30.0, size=(200, 1))

    X_train = np.hstack([
        normal_speeds, max_speeds, stop_ratios, distances,
        dir_changes, curvatures, zone_trans, durations
    ])

    print(f"Training IsolationForest on {len(X_train)} normal trajectory vectors...")
    model = IsolationForest(contamination=0.10, random_state=42, n_estimators=100)
    model.fit(X_train)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[SUCCESS] Pre-trained anomaly model saved to: {output_path}")

if __name__ == "__main__":
    train_baseline_model()
