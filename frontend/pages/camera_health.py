import streamlit as st
import time

st.set_page_config(page_title="Camera Health — IBVAP-X", page_icon="🎥", layout="wide")

st.title("🎥 Camera Health & Reliability Engine")
st.caption("Independent observation quality evaluation: Blur, Brightness, Frame Continuity & Obstruction")

st.markdown("---")

# Demo camera selector
cam_id = st.sidebar.selectbox("Select Camera", ["CAM-01", "CAM-02", "CAM-03"])
demo_degraded = st.sidebar.checkbox("⚡ Simulate Camera Degradation (Blur & Dark)", value=False)

col1, col2, col3, col4 = st.columns(4)

if demo_degraded:
    rel_score = 42.0
    status_str = "DEGRADED"
    status_color = "orange"
    reasons = [
        "High image blur detected (Laplacian variance 14.2 < threshold 50.0)",
        "Low brightness/underexposure detected (Mean: 28.5 < min 40.0)"
    ]
    blur_score = 30.0
    brightness_score = 35.0
    frame_score = 100.0
    obstruction_score = 60.0
else:
    rel_score = 94.5
    status_str = "GOOD"
    status_color = "green"
    reasons = []
    blur_score = 95.0
    brightness_score = 98.0
    frame_score = 100.0
    obstruction_score = 85.0

with col1:
    st.metric(label=f"Composite Reliability ({cam_id})", value=f"{rel_score}%", delta=status_str)
with col2:
    st.metric(label="Sharpness / Blur", value=f"{blur_score}%")
with col3:
    st.metric(label="Lighting Intensity", value=f"{brightness_score}%")
with col4:
    st.metric(label="Frame Continuity", value=f"{frame_score}%")

st.markdown("---")

st.subheader("📊 Reliability Sub-Metric Breakdown")
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.write("**Blur / Sharpness Score:**")
    st.progress(int(blur_score) / 100)
    st.write("**Brightness / Exposure Score:**")
    st.progress(int(brightness_score) / 100)

with col_b2:
    st.write("**Frame Health & Continuity:**")
    st.progress(int(frame_score) / 100)
    st.write("**Obstruction Heuristic Score:**")
    st.progress(int(obstruction_score) / 100)

st.markdown("---")

st.subheader("🔍 Active Degradation Reasons")
if reasons:
    for r in reasons:
        st.warning(f"⚠️ {r}")
else:
    st.success("✅ Camera feed is clear and performing within normal operational parameters.")
