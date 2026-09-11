#!/usr/bin/env python3
"""
IBVAP-X Synthetic Test Video Generator
Generates 3 synthetic test videos using OpenCV primitives for deterministic unit/math testing:
- test_normal.mp4   (shape moving smoothly across screen)
- test_zone.mp4     (shape entering a specified zone area)
- test_degraded.mp4 (shape moving in blurred/darkened scene)
"""

import os
import cv2
import numpy as np

def generate_test_video(output_path: str, duration_sec: int = 5, fps: int = 30, scenario: str = "normal"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    print(f"Generating synthetic test video: {output_path} (Scenario: {scenario})...")

    # Defined zone polygon in pixel space for test_zone
    zone_pts = np.array([[200, 150], [450, 150], [450, 350], [200, 350]], np.int32)

    for i in range(total_frames):
        # Create background canvas
        if scenario == "degraded":
            canvas = np.zeros((height, width, 3), dtype=np.uint8) + 30  # Dark background
        else:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 200  # Normal light gray

        # Draw test zone on scenario == "zone" or "normal"
        if scenario in ["zone", "normal"]:
            cv2.polylines(canvas, [zone_pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Calculate trajectory for test object (rectangle)
        progress = i / float(total_frames)
        if scenario == "zone":
            # Start outside zone, move into center of zone and remain
            x = int(50 + progress * 300)
            y = int(250)
        else:
            # Linear horizontal movement across screen
            x = int(50 + progress * 500)
            y = int(200 + np.sin(progress * np.pi * 2) * 50)

        # Draw synthetic moving target object (simulated person/object)
        cv2.rectangle(canvas, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
        cv2.putText(canvas, f"Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Apply degradation if scenario == "degraded"
        if scenario == "degraded":
            canvas = cv2.GaussianBlur(canvas, (31, 31), 0)

        out.write(canvas)

    out.release()
    print(f"  [OK] Created {output_path} ({total_frames} frames)")

def main():
    base_dir = os.path.join("data", "demo")
    generate_test_video(os.path.join(base_dir, "test_normal.mp4"), duration_sec=3, scenario="normal")
    generate_test_video(os.path.join(base_dir, "test_zone.mp4"), duration_sec=3, scenario="zone")
    generate_test_video(os.path.join(base_dir, "test_degraded.mp4"), duration_sec=3, scenario="degraded")
    print("Synthetic video generation complete.")

if __name__ == "__main__":
    main()
