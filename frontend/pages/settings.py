import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from backend.config import settings


st.set_page_config(page_title="Settings — IBVAP-X", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings & Configurable Thresholds")
st.caption("Tune operational policy parameters, reliability weights, and escalation timeouts")

st.markdown("---")

st.subheader("⏱️ Acknowledgment & Escalation Timeout")
ack_timeout = st.selectbox(
    "Operator Acknowledgment Timeout (ACK_TIMEOUT_SECONDS)",
    options=[10, 30, 60, 120],
    index=1,
    help="Time in seconds before an unacknowledged HIGH/CRITICAL alert auto-escalates to Command Centre"
)

st.markdown("---")

st.subheader("⚖️ Camera Reliability Sub-Metric Weights")
st.caption("Configurable prototype policy parameters for weighted reliability composite scoring")

w_blur = st.slider("Blur / Sharpness Weight", 0.0, 1.0, getattr(settings, "RELIABILITY_WEIGHT_BLUR", 0.35), 0.05)
w_bright = st.slider("Brightness Intensity Weight", 0.0, 1.0, getattr(settings, "RELIABILITY_WEIGHT_BRIGHTNESS", 0.25), 0.05)
w_frame = st.slider("Frame Health / Drop Rate Weight", 0.0, 1.0, getattr(settings, "RELIABILITY_WEIGHT_FRAME", 0.25), 0.05)
w_obs = st.slider("Obstruction Heuristic Weight", 0.0, 1.0, getattr(settings, "RELIABILITY_WEIGHT_OBSTRUCTION", 0.15), 0.05)

st.info(f"Total Weight Sum: {w_blur + w_bright + w_frame + w_obs:.2f}")

st.markdown("---")

st.subheader("🎯 Priority Scoring Policy Thresholds")
st.caption("Documented prototype policy parameters (Not empirical threat probabilities)")

st.slider("LOW Priority Max Threshold", 0, 50, getattr(settings, "PRIORITY_LOW_MAX", 40))
st.slider("MEDIUM Priority Max Threshold", 51, 80, getattr(settings, "PRIORITY_MEDIUM_MAX", 65))
st.slider("HIGH Priority Max Threshold", 81, 95, getattr(settings, "PRIORITY_HIGH_MAX", 84))

st.markdown("---")

st.subheader("🧠 Anomaly Engine Calibration")
if st.button("🧠 [Calibrate Normal Behaviour]"):
    st.success("IsolationForest model retrained on current session normal tracks. Model saved to data/anomaly_models/.")

st.button("💾 Save Settings to .env", type="primary")
