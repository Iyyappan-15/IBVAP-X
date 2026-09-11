#!/usr/bin/env python3
"""
IBVAP-X YOLO Model Downloader
Downloads yolov8n.pt to data/models/ if not already cached locally.
"""

import os
import sys
import argparse

def download_model(model_name: str = "yolov8n.pt", target_dir: str = "data/models", check_only: bool = False):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)

    if os.path.exists(target_path):
        print(f"[OK] YOLO model weights already present at: {target_path}")
        return True

    if check_only:
        print(f"[MISSING] YOLO model weights missing at: {target_path}")
        return False

    print(f"Downloading YOLO model weights '{model_name}' to {target_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        # Save model weights to target directory
        if hasattr(model, "model"):
            model.save(target_path)
        print(f"[SUCCESS] Downloaded and saved model weights to: {target_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download YOLO model weights: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download YOLO model weights for IBVAP-X")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--dir", type=str, default="data/models", help="Target directory")
    parser.add_argument("--check-only", action="store_true", help="Only check if model exists")

    args = parser.parse_args()
    success = download_model(model_name=args.model, target_dir=args.dir, check_only=args.check_only)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
