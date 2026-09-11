#!/usr/bin/env python3
"""
IBVAP-X Seed Demo Data Script
Seeds PostgreSQL / SQLite database with demo users, camera configurations, sample alerts, and evidence metadata.
"""

import sys
import os
import time

from backend.db.session import SessionLocal, init_db
from backend.db.repository import Repository
from backend.auth.auth import hash_password
from backend.interfaces import UserRole, CameraStatus, EventPriority, Actionability, AlertState, AlertOutput, ReliabilityScore

def seed_data():
    init_db()
    db = SessionLocal()
    repo = Repository(db)

    print("=== Seeding IBVAP-X Demo Data ===")

    # 1. Seed Demo Users
    users_data = [
        ("bop_operator", "demo1234", UserRole.BOP_OPERATOR),
        ("cmd_operator", "demo5678", UserRole.COMMAND_OPERATOR),
        ("admin", "admin9012", UserRole.ADMIN),
    ]

    for username, password, role in users_data:
        existing = repo.get_user_by_username(username)
        if not existing:
            hashed = hash_password(password)
            repo.create_user(username=username, hashed_password=hashed, role=role)
            print(f"  [USER] Created user '{username}' (Role: {role.value})")
        else:
            print(f"  [USER] User '{username}' already exists.")

    # 2. Seed Demo Cameras
    cams_data = [
        ("CAM-01", "North Boundary Gate - Alpha", "Sector 4 North Perimeter Fence", 31.6215, 74.8752, 45.0, 65.0, 150.0),
        ("CAM-02", "Central Ridge Watchtower", "Sector 4 Central Elevated Post", 31.6230, 74.8780, 90.0, 70.0, 180.0),
        ("CAM-03", "East Riverine Outpost", "Sector 4 Riverine Border Marker 12", 31.6260, 74.8820, 135.0, 60.0, 140.0),
    ]

    for cid, name, loc, lat, lon, heading, fov, range_m in cams_data:
        repo.save_camera(cid, name, loc, lat, lon, heading, fov, range_m)
        print(f"  [CAMERA] Saved camera '{cid}' ({name})")

        # Save initial health snapshot
        rel_score = ReliabilityScore(
            camera_id=cid,
            timestamp=time.time(),
            blur_score=90.0,
            brightness_score=95.0,
            frame_health_score=100.0,
            obstruction_score=90.0,
            composite_reliability_score=93.5,
            status=CameraStatus.GOOD,
            reasons=[]
        )
        repo.save_camera_health(rel_score)

    # 3. Seed Sample Historical Alerts
    now = time.time()
    sample_alert_1 = AlertOutput(
        alert_id="ALT-8F92A1",
        camera_id="CAM-03",
        track_id=17,
        class_name="person",
        timestamp=now - 300.0,
        event_priority=EventPriority.HIGH,
        event_priority_score=78.0,
        camera_reliability=CameraStatus.DEGRADED,
        camera_reliability_score=42.0,
        actionability=Actionability.MEDIUM,
        action_recommendation="High priority event observed on DEGRADED camera feed. Verify camera condition before escalation.",
        why_reasons=[
            "Restricted zone entry detected (Restricted Waterway Delta)",
            "Night-time operational context event (NIGHT)",
            "Sustained loitering detected (43s > threshold)",
            "Trajectory movement vector directed toward border boundary (TOWARD_BOUNDARY)"
        ],
        state=AlertState.ACTIVE,
        correlated_cameras=["CAM-01", "CAM-02"]
    )
    repo.save_alert(sample_alert_1)
    print("  [ALERT] Seeded sample alert 'ALT-8F92A1'")

    db.close()
    print("=== Demo Data Seeding Complete ===")

if __name__ == "__main__":
    seed_data()
