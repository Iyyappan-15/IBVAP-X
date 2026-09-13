"""
Standalone CLI tool to benchmark Baseline (YOLOv8n) vs Upgraded (YOLOv8s) vs Hybrid (YOLO + Open-Vocab) detection modes.
Usage: python scripts/benchmark_detection_modes.py --video data/demo/cctv_night_patrol.mp4
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.interfaces import Frame
from backend.detection.video_stream import FileVideoSource
from backend.pipeline import IBVAPXPipeline

logger = logging.getLogger("benchmark")

def run_benchmark_mode(video_path: str, mode_name: str, max_frames: int = 100) -> dict:
    print(f"\n[*] Running Benchmark Mode: {mode_name} ...")
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_start_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_start_mb = 0.0

    # Configure detection mode settings
    if "Baseline" in mode_name:
        settings.HYBRID_DETECTION_MODE = "standard"
        settings.YOLO_MODEL_PATH = "data/models/yolov8n.pt"
    elif "Upgraded" in mode_name:
        settings.HYBRID_DETECTION_MODE = "standard"
        up_path = "data/models/yolov8s.pt"
        if not os.path.exists(up_path):
            try:
                from ultralytics import YOLO
                print(f"[*] Downloading YOLOv8s weights to {up_path} ...")
                m = YOLO("yolov8s.pt")
                if os.path.exists("yolov8s.pt"):
                    import shutil
                    shutil.move("yolov8s.pt", up_path)
            except Exception:
                up_path = "data/models/yolov8n.pt"
        settings.YOLO_MODEL_PATH = up_path
    else:  # Hybrid
        settings.HYBRID_DETECTION_MODE = "hybrid"
        settings.YOLO_MODEL_PATH = "data/models/yolov8n.pt"

    source = FileVideoSource(file_path=video_path, camera_id="CAM-BENCHMARK")
    pipeline = IBVAPXPipeline()

    if "Hybrid" in mode_name and hasattr(pipeline, "open_vocab_engine") and pipeline.open_vocab_engine.is_available:
        pipeline.open_vocab_engine.set_classes([
            "person", "dog", "vehicle", "fence", "gate", "rock", "stone", "border marker"
        ])

    frame_count = 0
    total_detections = 0
    open_vocab_discoveries = 0
    tracks_seen = set()
    label_stabilities = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    start_time = time.time()

    while source.is_connected and frame_count < max_frames:
        frame_obj = source.get_frame()
        if frame_obj is None:
            break
            
        nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        frame_count += 1
        annotated, new_alerts = pipeline.process_frame(source, frame_obj, img)
        
        dets = getattr(pipeline, "last_detections", [])
        total_detections += len(dets)
        
        tracks = getattr(pipeline, "last_tracks", [])
        for t in tracks:
            tracks_seen.add(t.track_id)
            stab = getattr(t, "label_stability", "HIGH")
            label_stabilities[stab] = label_stabilities.get(stab, 0) + 1
            if getattr(t, "source", "") == "OPEN_VOCAB_DISCOVERY":
                open_vocab_discoveries += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0

    try:
        mem_end_mb = process.memory_info().rss / (1024 * 1024)
        mem_delta_mb = max(0.0, mem_end_mb - mem_start_mb)
    except Exception:
        mem_delta_mb = 0.0

    source.release()

    return {
        "mode": mode_name,
        "frames_evaluated": frame_count,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(fps, 1),
        "total_detections": total_detections,
        "unique_tracks": len(tracks_seen),
        "open_vocab_discoveries": open_vocab_discoveries,
        "label_stabilities": label_stabilities,
        "memory_mb": round(mem_delta_mb, 1)
    }

def print_benchmark_matrix(results: list):
    print("\n" + "=" * 95)
    print("📊 IBVAP-X SYSTEM BENCHMARK MATRIX — COMPARATIVE DETECTION PERFORMANCE")
    print("=" * 95)
    print(f"{'Mode':<32} | {'FPS':<6} | {'Frames':<7} | {'Detections':<11} | {'Tracks':<7} | {'OV Discoveries':<14} | {'Memory (MB)':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['mode']:<32} | {r['fps']:<6.1f} | {r['frames_evaluated']:<7} | {r['total_detections']:<11} | {r['unique_tracks']:<7} | {r['open_vocab_discoveries']:<14} | {r['memory_mb']:<10.1f}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IBVAP-X detection modes.")
    parser.add_argument("--video", default="data/demo/cctv_night_patrol.mp4", help="Path to test video file")
    parser.add_argument("--max-frames", type=int, default=80, help="Number of frames to benchmark per mode")
    args = parser.parse_args()

    video_file = args.video
    if not os.path.exists(video_file):
        print(f"[!] Test video not found at {video_file}. Creating synthetic test video...")
        os.makedirs(os.path.dirname(os.path.abspath(video_file)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file, fourcc, 25.0, (640, 480))
        for i in range(100):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (200, 300), (0, 255, 0), -1)
            out.write(frame)
        out.release()

    modes = ["1. Baseline (YOLOv8n)", "2. Upgraded (YOLOv8s)", "3. Hybrid (YOLO + Open-Vocab)"]
    results = []
    for mode in modes:
        res = run_benchmark_mode(video_file, mode, max_frames=args.max_frames)
        results.append(res)

    print_benchmark_matrix(results)
