"""
Root Streamlit Entrypoint for Cloud & Local Deployment.
Ensures repository root is in sys.path and delegates directly to frontend/dashboard.py.
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import runpy
runpy.run_path(os.path.join(ROOT_DIR, "frontend", "dashboard.py"), run_name="__main__")
