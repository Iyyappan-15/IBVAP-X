#!/usr/bin/env python3
"""
IBVAP-X Compatibility Verification Script
Verifies that all core packages import cleanly in the current Python environment.
"""

import sys

def check_environment():
    print(f"=== IBVAP-X Environment Check ===")
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]} (Target: 3.13 / 3.14 compatible)")
    print("-" * 50)

    modules_to_check = [
        ("ultralytics", "YOLO Object Detection"),
        ("supervision", "ByteTrack & Supervision Utilities"),
        ("shapely", "Geometric FOV & Polygon Computations"),
        ("folium", "Geospatial Map Visualization"),
        ("psycopg", "PostgreSQL Database Driver (v3)"),
        ("sqlite3", "SQLite Database Fallback Driver"),
        ("sklearn", "IsolationForest Anomaly Engine"),
        ("streamlit", "Operator Dashboard UI"),
        ("fastapi", "REST API Framework"),
        ("uvicorn", "ASGI Server"),
        ("cv2", "OpenCV Image Processing"),
        ("pydantic_settings", "Configuration Settings"),
        ("sqlalchemy", "Database ORM Engine"),
        ("argon2", "Password Hashing (Argon2id)"),
    ]

    all_passed = True

    for mod_name, label in modules_to_check:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "Available")
            print(f"  [OK]   {mod_name:<18} ({version:<12}) - {label}")
        except ImportError as e:
            print(f"  [FAIL] {mod_name:<18} NOT FOUND - {label} ({e})")
            all_passed = False

    print("-" * 50)
    if all_passed:
        print("[SUCCESS] All core modules imported successfully. Environment is valid.")
        return 0
    else:
        print("[ERROR] One or more core dependencies are missing. Run pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(check_environment())
