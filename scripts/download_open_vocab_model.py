"""
Script to explicitly download and cache Open-Vocabulary YOLO-World v2 model weights.
Usage: python scripts/download_open_vocab_model.py [--target-dir data/models]
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(target_dir: str = "data/models", model_name: str = "yolov8s-worldv2.pt") -> str:
    target_path = Path(target_dir) / model_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[+] Model checkpoint already exists at {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB)")
        return str(target_path)
        
    print(f"[*] Downloading {model_name} using Ultralytics...")
    try:
        from ultralytics import YOLOWorld
        model = YOLOWorld(model_name)
        # Ultralytics downloads to cwd or weights dir; locate downloaded file
        downloaded_pt = Path(model_name)
        if downloaded_pt.exists():
            import shutil
            shutil.move(str(downloaded_pt), str(target_path))
            print(f"[+] Successfully moved model to {target_path}")
        else:
            print(f"[+] Model successfully downloaded and cached.")
        return str(target_path)
    except Exception as e:
        print(f"[!] Failed to download {model_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLO-World model weights for IBVAP-X")
    parser.add_argument("--target-dir", default="data/models", help="Directory to store model weights")
    parser.add_argument("--model-name", default="yolov8s-worldv2.pt", help="YOLO-World model checkpoint name")
    args = parser.parse_args()
    
    download_model(target_dir=args.target_dir, model_name=args.model_name)
