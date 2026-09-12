"""
frontend/dashboard.py

IBVAP-X — Border Video Intelligence Operations Dashboard
Central Command & Decision Support Center for Border Surveillance.

Smart India Hackathon Problem Statement ID: 26187
AI-Based Intelligent Video Analytics Platform for Border Surveillance using existing CCTV Infrastructure.
"""
from __future__ import annotations


import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st


st.set_page_config(
    page_title="IBVAP-X — Border Video Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Military / Command Center Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 24px 30px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .badge-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-primary { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    .badge-success { background-color: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .badge-warning { background-color: #78350f; color: #fde68a; border: 1px solid #f59e0b; }
    .card-box {
        background: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HERO COMMAND HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span class="badge-chip badge-primary">DEFENSE AI PROTOCOL</span>
                <span class="badge-chip badge-success">SIH PS #26187</span>
                <span class="badge-chip badge-warning">OPERATIONAL READY</span>
                <h1 style="color: #f8fafc; margin: 10px 0 4px 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px;">
                    🛡️ IBVAP-X Operations Command Center
                </h1>
                <p style="color: #94a3b8; margin: 0; font-size: 15px;">
                    <strong>Reliability-Aware Border Video Intelligence</strong> — Transforming legacy CCTV feeds into an autonomous, explainable surveillance network.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE OPERATIONAL METRICS BAR
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="📹 Active Camera Nodes",
        value="4 Channels",
        delta="3 Perimeter + 1 Upload",
        help="Monitored border sectors: Alpha, Bravo, Delta, and Dynamic Upload Stream.",
    )

with m2:
    st.metric(
        label="📡 Network Reliability",
        value="96.2%",
        delta="🟢 GOOD Condition",
        help="Composite real-time score computed from Blur, Brightness, Lens Obstruction & Frame Health.",
    )

with m3:
    st.metric(
        label="🚨 Active Intelligence Alerts",
        value="2 Queued",
        delta="1 High Priority",
        help="Priority-filtered events cross-referenced with spatial boundary context.",
    )

with m4:
    st.metric(
        label="🔒 Evidence Storage Ledger",
        value="SHA-256",
        delta="100% Tamper-Evident",
        help="All high/critical alert evidence clips are cryptographically signed with Argon2id + SHA-256 hash chains.",
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CORE INNOVATION: THE 3-SIGNAL DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🧠 The Core Breakthrough: The 3-Signal Decision Architecture")
st.caption("Why IBVAP-X solves false alarms in border CCTV where traditional AI systems fail:")

c_sig1, c_sig2, c_sig3 = st.columns(3)

with c_sig1:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="color: #f87171; margin-top:0;">🚨 1. Event Threat Priority (0–100)</h4>
            <p style="color: #cbd5e1; font-size: 14px;">
                Measures the tactical severity of the observed action based on <strong>Spatial Boundary Zones</strong>, <strong>Nocturnal Context</strong>, <strong>Loitering Duration</strong>, and <strong>Movement Trajectory Vectors</strong> directed toward the borderline.
            </p>
            <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 12px; color: #94a3b8;">
                ✓ Rule-based transparent scoring<br>
                ✓ Explainable reasons per alert<br>
                ✓ Multi-factor priority tiers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_sig2:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="color: #38bdf8; margin-top:0;">📡 2. Camera Reliability Index (0–100%)</h4>
            <p style="color: #cbd5e1; font-size: 14px;">
                Parallel intelligence stream independently evaluating feed degradation: <strong>Laplacian Blur</strong> (out-of-focus), <strong>Mean Luminance</strong> (darkness/glare), <strong>Frame Drop Glitches</strong>, and <strong>Physical Lens Obstruction</strong>.
            </p>
            <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 12px; color: #94a3b8;">
                ✓ Real-time parallel stream<br>
                ✓ Independent of detection boxes<br>
                ✓ Automatic degradation flags
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_sig3:
    st.markdown(
        """
        <div class="card-box">
            <h4 style="color: #4ade80; margin-top:0;">🎯 3. Actionability Matrix</h4>
            <p style="color: #cbd5e1; font-size: 14px;">
                Synthesizes Threat Priority with Camera Reliability. If a high-priority event is detected on a <strong>DEGRADED</strong> camera, the system automatically marks it <strong>VERIFY CAMERA FIRST</strong> rather than mobilizing tactical squads for a false ghost alarm.
            </p>
            <div style="background: #0f172a; padding: 8px 12px; border-radius: 6px; font-size: 12px; color: #94a3b8;">
                ✓ High Priority + Good Cam → IMMEDIATE ESCALATE<br>
                ✓ High Priority + Degraded → VERIFY FEED FIRST<br>
                ✓ Low Priority + Poor Cam → LOG AUDIT ONLY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# QUICK LAUNCHPAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🚀 Module Navigation Launchpad")

lp1, lp2, lp3, lp4 = st.columns(4)

with lp1:
    st.markdown("#### 🎥 Live Monitoring")
    st.write("Upload video files (MP4/AVI/MOV), test demo feeds, and inspect real-time detection & tracking.")
    if st.button("▶️ Launch Live Monitoring", key="lp_btn_live", type="primary", use_container_width=True):
        st.switch_page("pages/live_monitoring.py")

with lp2:
    st.markdown("#### 🚨 Operational Alerts")
    st.write("Review prioritized border threat alerts with explainability breakdowns and operator ACK controls.")
    if st.button("📋 Open Alerts Queue", key="lp_btn_alerts", use_container_width=True):
        st.switch_page("pages/alerts.py")

with lp3:
    st.markdown("#### 🗺️ Coverage Gap Map")
    st.write("Explore the geometric FOV sector map, blind-spot zones, and camera overlap regions.")
    if st.button("🌐 Open Sector Map", key="lp_btn_map", use_container_width=True):
        st.switch_page("pages/coverage_map.py")

with lp4:
    st.markdown("#### 🔒 Evidence Locker")
    st.write("Audit cryptographically signed video evidence clips with interactive SHA-256 tamper verification.")
    if st.button("🛡️ Open Evidence Locker", key="lp_btn_ev", use_container_width=True):
        st.switch_page("pages/evidence.py")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# BORDER CAMERA SECTOR DEPLOYMENT STATUS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📡 Deployed Border Camera Network Status")

camera_status_data = [
    {
        "Channel": "CAM-01",
        "Sector Name": "North Boundary Gate - Alpha",
        "Coordinates": "31.6215° N, 74.8752° E",
        "FOV Angle": "65.0° (Heading 45°)",
        "Range": "150 meters",
        "Reliability Score": "94% [GOOD]",
        "Active Policy Zone": "Restricted Sector Alpha",
        "Status": "🟢 ONLINE",
    },
    {
        "Channel": "CAM-02",
        "Sector Name": "Central Ridge Watchtower",
        "Coordinates": "31.6230° N, 74.8780° E",
        "FOV Angle": "70.0° (Heading 90°)",
        "Range": "180 meters",
        "Reliability Score": "92% [GOOD]",
        "Active Policy Zone": "Buffer Zone Bravo",
        "Status": "🟢 ONLINE",
    },
    {
        "Channel": "CAM-03",
        "Sector Name": "East Riverine Outpost",
        "Coordinates": "31.6260° N, 74.8820° E",
        "FOV Angle": "60.0° (Heading 135°)",
        "Range": "140 meters",
        "Reliability Score": "42% [DEGRADED]",
        "Active Policy Zone": "Restricted Waterway Delta",
        "Status": "🟡 DEGRADED",
    },
    {
        "Channel": "CAM-UPLOAD-01",
        "Sector Name": "Dynamic Video Ingestion Channel",
        "Coordinates": "31.6240° N, 74.8765° E",
        "FOV Angle": "70.0° (Heading 90°)",
        "Range": "150 meters",
        "Reliability Score": "100% [ONLINE]",
        "Active Policy Zone": "Upload Primary Zone",
        "Status": "🟢 READY",
    },
]

st.dataframe(camera_status_data, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM ARCHITECTURE & COMPONENT BLUEPRINT
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🏗️ View Full IBVAP-X Intelligence Pipeline Architecture", expanded=False):
    st.markdown(
        """
```
Legacy CCTV Video Feed (File / Webcam / RTSP)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: Video Ingestion & Resolution Preprocessing     │
└──────────────────────────┬───────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌───────────────────────────────┐
│ Stage 2: Object Detector│ │ Parallel Stream A:            │
│ YOLOv8n + Sub-box Filter│ │ Camera Reliability Engine     │
│ + Perimeter Fence Model │ │ Blur, Brightness, Obstruction │
└────────────┬────────────┘ └──────────────┬────────────────┘
             ▼                             │
┌─────────────────────────┐                │
│ Stage 3: ByteTrack      │                │
│ Kalman Tracking Engine  │                │
└────────────┬────────────┘                │
             ▼                             │
┌─────────────────────────┐                │
│ Stage 4: Context Engine │                │
│ Zones, Time, Loitering  │                │
└────────────┬────────────┘                │
             ▼                             │
┌─────────────────────────┐                │
│ Stage 5: Anomaly Engine │                │
│ Isolation Forest Model  │                │
└────────────┬────────────┘                │
             ▼                             │
┌─────────────────────────┐                │
│ Stage 6: Cross-Camera   │                │
│ Spatial Correlation     │                │
└────────────┬────────────┘                │
             │                             │
             └──────────────┬──────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 7: Priority Scoring & Actionability Matrix         │
│ (Independent Threat Priority vs Feed Reliability)        │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 8: Explainable Alert Queue & Decision Dispatch     │
└───────────────────────────┬──────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 9: Tamper-Evident Evidence Vault (SHA-256 Hashes)  │
└──────────────────────────────────────────────────────────┘
```
        """
    )

st.caption(
    "IBVAP-X Prototype · Built for Smart India Hackathon PS ID: 26187 · "
    "Full-Stack Defense Video Analytics Platform"
)
