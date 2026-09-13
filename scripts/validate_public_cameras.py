"""
CLI Tool to Validate Public Camera Registry & External Streams for IBVAP-X.
Measures latency, stream status, and initial frame acquisition without mutating static JSON configuration files.
"""

import sys
import os
import json
import time
import argparse
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from backend.detection.public_camera import PublicCameraSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_public_cameras")


def validate_public_cameras(config_path: str):
    if not os.path.exists(config_path):
        logger.error(f"Public camera configuration file not found at: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cameras = data.get("public_cameras", [])
    logger.info(f"Loaded {len(cameras)} public camera configuration entries from '{config_path}'.")

    print("\n" + "=" * 95)
    print(f"{'ID':<15} | {'NAME':<30} | {'TYPE':<10} | {'STATUS':<10} | {'LATENCY (ms)':<12} | {'RES'}")
    print("=" * 95)

    online_count = 0
    offline_count = 0
    view_only_count = 0

    for cam in cameras:
        cam_id = cam.get("camera_id", "N/A")
        name = cam.get("name", "Unknown")[:30]
        source_type = cam.get("source_type", "UNKNOWN")
        enabled = cam.get("enabled", True)

        if not enabled:
            print(f"{cam_id:<15} | {name:<30} | {source_type:<10} | {'DISABLED':<10} | {'N/A':<12} | N/A")
            continue

        if source_type == "VIEW_ONLY":
            view_only_count += 1
            print(f"{cam_id:<15} | {name:<30} | {source_type:<10} | {'VIEW_ONLY':<10} | {'N/A':<12} | Web View")
            continue

        t0 = time.time()
        source = PublicCameraSource(cam)
        latency_ms = (time.time() - t0) * 1000.0

        if source.is_connected:
            frame = source.get_frame()
            if frame is not None:
                online_count += 1
                res_str = f"{source.width}x{source.height}"
                print(f"{cam_id:<15} | {name:<30} | {source_type:<10} | {'ONLINE':<10} | {latency_ms:<12.1f} | {res_str}")
            else:
                offline_count += 1
                print(f"{cam_id:<15} | {name:<30} | {source_type:<10} | {'DEGRADED':<10} | {latency_ms:<12.1f} | Frame Drop")
        else:
            offline_count += 1
            print(f"{cam_id:<15} | {name:<30} | {source_type:<10} | {'OFFLINE':<10} | {latency_ms:<12.1f} | Fail")

        source.release()

    print("=" * 95)
    print(f"Summary: {online_count} Online | {offline_count} Offline | {view_only_count} View-Only | Total: {len(cameras)}")
    print("=" * 95 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate IBVAP-X Public Camera Stream Registry")
    parser.add_argument(
        "--config",
        type=str,
        default=settings.PUBLIC_CAMERAS_CONFIG_PATH,
        help="Path to public cameras json file",
    )
    args = parser.parse_args()
    validate_public_cameras(args.config)
