import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import time

from backend.detection.video_stream import FileVideoSource, DemoVideoSource
from backend.pipeline import IBVAPXPipeline

st.set_page_config(page_title="Live Monitoring — IBVAP-X", page_icon="🎥", layout="wide")

st.title("🎥 Live Video Monitoring & Intelligence Stream")
st.caption("Primary Prototype Input Workflow: Upload Video, Webcam, RTSP, or Demo Stream")

st.markdown("---")

# Video Source Selector (Refinement 1 from approved plan)
st.sidebar.header("📹 Video Input Source")
input_type = st.sidebar.radio(
    "Select Video Source Type",
    ["Upload Video File (MP4/AVI)", "Demo Video Stream", "Webcam Device", "RTSP Network Stream"]
)

demo_degraded = st.sidebar.checkbox("⚡ Simulate Camera Degradation (Blur & Dark)", value=False)

selected_file_path = None

if input_type == "Upload Video File (MP4/AVI)":
    uploaded_file = st.sidebar.file_uploader("Upload Video Clip", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            selected_file_path = tfile.name
    else:
        st.sidebar.info("Using default sample video 'data/demo/test_normal.mp4'")
        selected_file_path = "data/demo/test_normal.mp4"

elif input_type == "Demo Video Stream":
    demo_options = {
        "Scenario 1: Normal Activity (CAM-01)": "data/demo/test_normal.mp4",
        "Scenario 2: Restricted Zone Entry (CAM-02)": "data/demo/test_zone.mp4",
        "Scenario 4: Degraded Camera Feed (CAM-03)": "data/demo/test_degraded.mp4"
    }
    selected_demo = st.sidebar.selectbox("Select Demo Scenario Video", list(demo_options.keys()))
    selected_file_path = demo_options[selected_demo]

col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("📺 Live Video Analytics Feed")
    video_placeholder = st.empty()

    if st.button("▶️ Start Video Pipeline Analysis"):
        if selected_file_path and os.path.exists(selected_file_path):
            source = FileVideoSource(file_path=selected_file_path, camera_id="CAM-01")
            pipeline = IBVAPXPipeline(enable_demo_degradation=demo_degraded)
            
            frame_count = 0
            while source.is_connected and frame_count < 60:
                frame_obj = source.get_frame()
                if frame_obj is None:
                    break

                frame_count += 1
                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                annotated, new_alerts = pipeline.process_frame(source, frame_obj, img)
                
                # Display annotated frame in Streamlit
                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_img, caption=f"Frame #{frame_obj.frame_id} (CAM-01)", use_container_width=True)
                time.sleep(0.05)

            source.release()
            st.success("Analysis complete.")
        else:
            st.error("No valid video file selected or file not found.")

with col_right:
    st.subheader("⚡ Feed Status")
    st.metric("Detection Engine", "YOLOv8n Active")
    st.metric("Tracking Engine", "ByteTrack / IoU Active")
    st.metric("Camera Reliability", "95.0%" if not demo_degraded else "42.0% (DEGRADED)")
    st.metric("Zone Context", "Sector Alpha Monitored")
