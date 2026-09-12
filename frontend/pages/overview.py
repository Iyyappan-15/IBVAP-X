import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

st.set_page_config(page_title="Overview — IBVAP-X", page_icon="📊", layout="wide")

st.title("📊 System Overview & Operational Command")
st.caption("Reliability-Aware Border Video Intelligence Dashboard")

st.markdown("---")

# Metrics summary bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Cameras Configured", "3", delta="3 Online")
with c2:
    st.metric("Avg Camera Reliability", "94.5%", delta="GOOD", delta_color="normal")
with c3:
    st.metric("Active High-Priority Alerts", "1", delta="Escalated to Command", delta_color="inverse")
with c4:
    st.metric("Evidence Chain Mode", "SHA-256", delta="Tamper-Evident Sealed")

st.markdown("---")

# 3-Signal Architecture Overview
st.subheader("🛡️ IBVAP-X 3-Signal Decision Architecture")
st.caption("The core innovation of IBVAP-X is evaluating Event Priority and Camera Reliability as two independent signals to determine actionable defense posture.")

s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    st.markdown(
        """<div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 14px;">
            <div style="font-size: 13px; font-weight: 700; color: #f87171;">SIGNAL 1: EVENT PRIORITY (0–100)</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px; line-height: 1.6;">
                Rule-based additive scoring based on spatial boundary entry (+45), night-time context (+20), loitering duration (+20), and movement vector toward border (+15).
            </div>
        </div>""",
        unsafe_allow_html=True
    )
with s_col2:
    st.markdown(
        """<div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 14px;">
            <div style="font-size: 13px; font-weight: 700; color: #4ade80;">SIGNAL 2: CAMERA RELIABILITY (0–100%)</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px; line-height: 1.6;">
                Continuous visual quality assessment across Laplacian blur (35%), illumination/brightness (25%), frame health (25%), and obstruction heuristics (15%).
            </div>
        </div>""",
        unsafe_allow_html=True
    )
with s_col3:
    st.markdown(
        """<div style="background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 14px;">
            <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">SIGNAL 3: ACTIONABILITY (4×4 MATRIX)</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 6px; line-height: 1.6;">
                Deterministic cross-product of Event Priority × Camera Reliability ensuring degraded cameras prompt camera verification before physical force dispatch.
            </div>
        </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📷 Monitored Camera Health Grid")
    
    cam_cols = st.columns(3)
    cams = [
        {"id": "CAM-01", "name": "North Gate Alpha", "status": "GOOD", "rel": 95.0, "location": "Perimeter Fence"},
        {"id": "CAM-02", "name": "Ridge Watchtower", "status": "GOOD", "rel": 98.0, "location": "Elevated Post"},
        {"id": "CAM-03", "name": "Riverine Outpost", "status": "DEGRADED", "rel": 42.0, "location": "Marker 12 (Low Light)"},
    ]

    for idx, c in enumerate(cams):
        with cam_cols[idx % 3]:
            badge = "🟢 GOOD" if c["status"] == "GOOD" else "🟠 DEGRADED"
            st.markdown(f"### {c['id']}")
            st.markdown(f"**{c['name']}**")
            st.markdown(f"Status: **{badge}** ({c['rel']}%)")
            st.caption(c["location"])
            st.progress(int(c["rel"]) / 100)

with col_right:
    st.subheader("⚡ Quick Navigation")
    if st.button("🎥 Open Live Monitoring & Video Stream", key="nav_live", use_container_width=True):
        st.switch_page("pages/live_monitoring.py")
    if st.button("🚨 View Operational Priority Alerts", key="nav_alerts", use_container_width=True):
        st.switch_page("pages/alerts.py")
    if st.button("📡 Camera Health & Diagnostics", key="nav_cam", use_container_width=True):
        st.switch_page("pages/camera_health.py")
    if st.button("🔒 Tamper-Evident Evidence Vault", key="nav_ev", use_container_width=True):
        st.switch_page("pages/evidence.py")

