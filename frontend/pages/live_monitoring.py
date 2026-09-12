"""
frontend/pages/live_monitoring.py

IBVAP-X — Live Video Monitoring & Intelligence Stream
Primary user-facing prototype page.

Implements the full Video Upload & Processing Addendum (27 points):
- Upload MP4/AVI/MOV → validate → show metadata → START ANALYSIS
- Frame-by-frame processing via full IBVAP-X pipeline (NEVER loads entire video into RAM)
- Progress bar, live stats, annotated frame display during processing
- Post-processing summary with navigation buttons
- STOP ANALYSIS button (clean shutdown)
- Camera Degradation Demo toggle (programmatic blur + darkness, no separate video needed)
- All errors shown as user-friendly messages (no Python stack traces)
- Source labelled "RECORDED VIDEO ANALYSIS / UPLOADED VIDEO"
- VideoSource abstraction preserved: FileVideoSource → same pipeline as RTSP future

Author: IBVAP-X Team
"""
from __future__ import annotations

import os
import time
import logging
import math

import cv2
import numpy as np
import streamlit as st

from backend.config.settings import settings
from backend.detection.video_stream import FileVideoSource, DemoVideoSource
from backend.pipeline import IBVAPXPipeline
from backend.upload.video_validator import validate_upload, resize_for_inference
from backend.upload.safe_temp_storage import (
    save_upload_to_temp,
    cleanup_temp_file,
    cleanup_old_uploads,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Monitoring — IBVAP-X",
    page_icon="🎥",
    layout="wide",
)

# One-time temp cleanup at page load (remove files older than 24 h)
try:
    cleanup_old_uploads(settings.VIDEO_TEMP_DIR)
except Exception:
    pass   # Never block the UI for cleanup failures

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎥 IBVAP-X — Live Video Monitoring & Intelligence Stream")
st.caption(
    "**Reliability-Aware Border Video Intelligence** · "
    "Upload a recorded video clip to run the full detection pipeline."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "ibvapx_stop_requested": False,
    "ibvapx_analysis_done": False,
    "ibvapx_summary": None,
    "ibvapx_temp_path": None,
    "ibvapx_validation_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — VIDEO INPUT SOURCE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📹 Video Input Source")
input_type = st.sidebar.radio(
    "Select Video Source Type",
    [
        "📤 Upload Video File (Recommended)",
        "🎬 Demo Video Stream",
        "📷 Webcam Device",
        "🌐 RTSP Network Stream",
    ],
)

st.sidebar.markdown("---")
demo_degraded = st.sidebar.checkbox(
    "⚡ Simulate Camera Degradation (Blur & Dark)",
    value=False,
    help=(
        "Applies programmatic blur + darkness to simulate a dirty/degraded camera lens. "
        "No separate blurry video file is needed."
    ),
)
st.sidebar.caption(
    "Camera Reliability Engine runs independently on all inputs, "
    "including uploaded videos (treated as a simulated camera feed)."
)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT-TYPE HANDLING
# ─────────────────────────────────────────────────────────────────────────────
selected_file_path: str | None = None
camera_id: str = settings.UPLOAD_CAMERA_ID
source_label: str = "UPLOADED VIDEO"

# ── A. UPLOAD ─────────────────────────────────────────────────────────────────
if "Upload" in input_type:
    st.markdown("### 📤 Upload Recorded Video")
    st.markdown(
        f"> **Recommended format:** MP4 (H.264) · "
        f"Max size: {settings.MAX_UPLOAD_SIZE_MB} MB · "
        f"Max duration: {settings.MAX_VIDEO_DURATION_SECONDS}s"
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="Supported: MP4 (H.264) — Recommended, AVI, MOV",
    )

    if uploaded_file is not None:
        # ── Pre-check file size before saving (fast reject before disk write) ──
        raw_bytes = uploaded_file.read()
        size_mb = len(raw_bytes) / (1024 * 1024)

        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            st.error(
                f"File is too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB} MB. "
                "Please trim or compress the video before uploading."
            )
        else:
            with st.spinner("Saving upload and validating video..."):
                try:
                    temp_path = save_upload_to_temp(
                        raw_bytes,
                        uploaded_file.name,
                        settings.VIDEO_TEMP_DIR,
                    )
                    st.session_state["ibvapx_temp_path"] = temp_path
                except Exception as e:
                    st.error(f"Could not save uploaded file. Please try again. ({e})")
                    temp_path = None

            if temp_path:
                vr = validate_upload(temp_path, uploaded_file.name)
                st.session_state["ibvapx_validation_result"] = vr
                st.session_state["ibvapx_analysis_done"] = False
                st.session_state["ibvapx_summary"] = None

                if vr.passed:
                    selected_file_path = temp_path
                    camera_id = settings.UPLOAD_CAMERA_ID
                    source_label = "RECORDED VIDEO ANALYSIS"

                    # ── Video Metadata Preview UI ──────────────────────────
                    st.success("✅ Video passed validation. Review metadata below, then click **START ANALYSIS**.")

                    with st.expander("📋 Video Metadata", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Filename", vr.filename)
                        c2.metric("File Size", f"{vr.file_size_mb:.1f} MB")
                        c3.metric("Duration", f"{vr.duration_seconds:.1f}s")

                        c4, c5, c6 = st.columns(3)
                        c4.metric("Resolution", f"{vr.width}x{vr.height}")
                        c5.metric("FPS", f"{vr.fps:.1f}")
                        c6.metric("Codec", vr.codec_fourcc or "N/A")

                        c7, c8 = st.columns(2)
                        c7.metric("Total Frames", vr.total_frames)
                        est = vr.estimated_process_time_seconds
                        c8.metric(
                            "Est. Processing Time",
                            f"{int(est)}s"
                            if est < 60
                            else f"{int(est // 60)}m {int(est % 60)}s",
                        )

                    for warn in vr.warnings:
                        st.warning(f"⚠️ {warn}")

                else:
                    cleanup_temp_file(temp_path)
                    st.session_state["ibvapx_temp_path"] = None
                    st.error(f"❌ Video failed validation: {vr.error_message}")
                    for warn in vr.warnings:
                        st.warning(f"⚠️ {warn}")
    else:
        st.info(
            "No file uploaded yet. "
            f"Accepted: MP4 (H.264), AVI, MOV — "
            f"Max {settings.MAX_UPLOAD_SIZE_MB} MB / {settings.MAX_VIDEO_DURATION_SECONDS}s."
        )

# ── B. DEMO ────────────────────────────────────────────────────────────────────
elif "Demo" in input_type:
    st.markdown("### 🎬 Demo Scenario Video")
    demo_options = {
        "Scenario 1: Normal Patrol Activity (CAM-01)": "data/demo/test_normal.mp4",
        "Scenario 2: Restricted Zone Entry (CAM-02)": "data/demo/test_zone.mp4",
        "Scenario 3: Degraded Camera Feed (CAM-03)": "data/demo/test_degraded.mp4",
    }
    selected_demo = st.selectbox("Select Demo Scenario", list(demo_options.keys()))
    demo_path = demo_options[selected_demo]

    if os.path.exists(demo_path):
        selected_file_path = demo_path
        camera_id = "CAM-01"
        source_label = "DEMO VIDEO STREAM"
        st.info(f"Demo video ready: `{demo_path}`")
    else:
        st.error(
            f"Demo video not found: `{demo_path}`. "
            "Run `python scripts/generate_test_videos.py` to create demo videos."
        )

# ── C. WEBCAM ─────────────────────────────────────────────────────────────────
elif "Webcam" in input_type:
    st.markdown("### 📷 Webcam Live Feed")
    st.info(
        "Webcam input is supported by the pipeline architecture (WebcamVideoSource). "
        "This interface uses uploaded video as the primary prototype input. "
        "To test webcam: select 'Upload Video File' and use a screen recording."
    )
    camera_id = "CAM-WEBCAM"
    source_label = "WEBCAM FEED"

# ── D. RTSP ────────────────────────────────────────────────────────────────────
elif "RTSP" in input_type:
    st.markdown("### 🌐 RTSP Network Stream")
    st.info(
        "RTSP support is built into the VideoSource abstraction (RTSPVideoSource). "
        "For this prototype demonstration, please use the Upload or Demo mode. "
        "RTSP streams can be connected by extending this page's source selector."
    )
    camera_id = "CAM-RTSP"
    source_label = "RTSP STREAM"

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
col_start, col_stop = st.columns([1, 1])

with col_start:
    start_clicked = st.button(
        "▶️ START ANALYSIS",
        type="primary",
        disabled=(selected_file_path is None),
        use_container_width=True,
    )

with col_stop:
    stop_clicked = st.button(
        "⏹ STOP ANALYSIS",
        type="secondary",
        use_container_width=True,
    )

if stop_clicked:
    st.session_state["ibvapx_stop_requested"] = True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS LOOP
# ─────────────────────────────────────────────────────────────────────────────
if start_clicked and selected_file_path:
    # Reset state
    st.session_state["ibvapx_stop_requested"] = False
    st.session_state["ibvapx_analysis_done"] = False
    st.session_state["ibvapx_summary"] = None

    source = None
    pipeline = None
    init_error = None

    try:
        source = FileVideoSource(file_path=selected_file_path, camera_id=camera_id)
        pipeline = IBVAPXPipeline(enable_demo_degradation=demo_degraded)
    except Exception as e:
        init_error = str(e)
        logger.error("Pipeline init error: %s", e)

    if init_error:
        st.error(f"Could not initialise the analysis pipeline. {init_error}")
    else:
        st.markdown(f"#### 📡 Analysis Feed  —  `{source_label}`  ·  Camera: `{camera_id}`")

        if demo_degraded:
            st.warning(
                "🎛️ **Camera Degradation Demo is ON** — "
                "Blur and darkness applied programmatically to simulate a dirty lens."
            )

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        stats_cols = st.columns(6)
        stat_frames  = stats_cols[0].empty()
        stat_fps     = stats_cols[1].empty()
        stat_tracks  = stats_cols[2].empty()
        stat_events  = stats_cols[3].empty()
        stat_hi_pri  = stats_cols[4].empty()
        stat_rel     = stats_cols[5].empty()

        total_frames     = source.total_frames
        source_fps       = source.fps if source.fps > 0 else 25.0
        process_interval = max(1, int(round(source_fps / settings.PROCESS_FPS)))
        frame_idx          = 0
        processed_count    = 0
        total_events       = 0
        high_priority_count = 0
        last_reliability_pct = 100.0
        last_tracked_count = 0
        # Track per-class counts for summary
        class_counts: dict = {}
        loop_start = time.time()

        try:
            while source.is_connected:
                if st.session_state.get("ibvapx_stop_requested", False):
                    st.warning("⏹ Analysis stopped by user. Partial results preserved.")
                    break

                frame_obj = source.get_frame()
                if frame_obj is None:
                    break

                frame_idx += 1
                if (frame_idx - 1) % process_interval != 0:
                    continue

                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                infer_img, was_resized = resize_for_inference(img)

                try:
                    annotated, new_alerts = pipeline.process_frame(source, frame_obj, infer_img)
                    if was_resized:
                        annotated = cv2.resize(
                            annotated,
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                except Exception as pipe_err:
                    logger.error("Pipeline frame error: %s", pipe_err)
                    annotated = img

                new_alerts = new_alerts or []
                processed_count += 1
                total_events += len(new_alerts)
                high_priority_count += sum(
                    1 for a in new_alerts
                    if hasattr(a, "event_priority")
                    and str(a.event_priority).upper() in ("HIGH", "CRITICAL")
                )

                # Tally detected classes for summary report
                try:
                    for trk in pipeline.tracker.fallback_tracker.active_tracks.values():
                        cname = trk.get("class_name", "unknown")
                        class_counts[cname] = class_counts.get(cname, 0) + 1
                except Exception:
                    pass

                # Reliability
                try:
                    rel_score = pipeline.reliability_engine.calculate_reliability(
                        camera_id=camera_id,
                        frame_id=frame_obj.frame_id,
                        timestamp=frame_obj.timestamp,
                        image_np=infer_img,
                    )
                    last_reliability_pct = rel_score.composite_score
                except Exception:
                    pass

                # Active track count
                try:
                    last_tracked_count = len(
                        pipeline.tracker.fallback_tracker.active_tracks
                    )
                except Exception:
                    pass

                # Source label overlay bar
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (20, 20, 20), -1)
                cv2.putText(
                    annotated,
                    f"[{source_label}]  Cam: {camera_id}  Frame: {frame_idx}/{total_frames}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 200), 1, cv2.LINE_AA,
                )

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    rgb,
                    caption=f"Frame {frame_idx}/{total_frames} — {source_label} | {camera_id}",
                    use_container_width=True,
                )

                progress_bar.progress(min(frame_idx / max(total_frames, 1), 1.0))

                elapsed = time.time() - loop_start
                live_fps = processed_count / elapsed if elapsed > 0 else 0.0
                stat_frames.metric("Frames", f"{frame_idx}/{total_frames}")
                stat_fps.metric("Processing FPS", f"{live_fps:.1f}")
                stat_tracks.metric("Tracked Objects", last_tracked_count)
                stat_events.metric("Events Detected", total_events)
                stat_hi_pri.metric("High/Critical", high_priority_count)
                rel_label = (
                    "GOOD"     if last_reliability_pct >= settings.RELIABILITY_GOOD_THRESHOLD
                    else "DEGRADED" if last_reliability_pct >= settings.RELIABILITY_DEGRADED_THRESHOLD
                    else "POOR"
                )
                stat_rel.metric("Camera Reliability", f"{last_reliability_pct:.0f}% [{rel_label}]")

                time.sleep(1.0 / settings.DISPLAY_FPS)

        except Exception as loop_err:
            logger.error("Analysis loop error: %s", loop_err)
            st.error(
                "An error occurred during analysis. Partial results have been preserved. "
                f"Error type: {type(loop_err).__name__}"
            )
        finally:
            if source:
                source.release()

        if not st.session_state.get("ibvapx_stop_requested", False):
            progress_bar.progress(1.0)
            st.success("✅ Analysis complete.")

        elapsed_total = time.time() - loop_start
        st.session_state["ibvapx_summary"] = {
            "source_label": source_label,
            "camera_id": camera_id,
            "total_frames": total_frames,
            "source_fps": round(source_fps, 2),
            "processed_frames": processed_count,
            "total_events": total_events,
            "high_priority_count": high_priority_count,
            "elapsed_seconds": elapsed_total,
            "reliability_pct": last_reliability_pct,
            "class_counts": dict(class_counts),
            "stopped_early": st.session_state.get("ibvapx_stop_requested", False),
        }
        st.session_state["ibvapx_analysis_done"] = True

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING SUMMARY UI  (Task 5 — full detailed report)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("ibvapx_analysis_done") and st.session_state.get("ibvapx_summary"):
    s = st.session_state["ibvapx_summary"]
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary Report")

    tag = "⚠️ STOPPED EARLY" if s["stopped_early"] else "✅ COMPLETE"
    rel_status = (
        "🟢 GOOD"     if s["reliability_pct"] >= settings.RELIABILITY_GOOD_THRESHOLD
        else "🟡 DEGRADED" if s["reliability_pct"] >= settings.RELIABILITY_DEGRADED_THRESHOLD
        else "🔴 POOR"
    )

    st.markdown(
        f"**Status:** {tag} &nbsp;|&nbsp; "
        f"**Source:** `{s['source_label']}` &nbsp;|&nbsp; "
        f"**Camera:** `{s['camera_id']}`"
    )

    # ── Metric row 1 — Processing stats ──────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Frames Analysed",  f"{s['processed_frames']}/{s['total_frames']}")
    m2.metric("Source FPS",       f"{s['source_fps']}")
    m3.metric("Events Detected",  s["total_events"])
    m4.metric("High/Critical",    s["high_priority_count"])
    elapsed = s["elapsed_seconds"]
    m5.metric(
        "Analysis Time",
        f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
    )
    m6.metric("Effective FPS", f"{s['processed_frames'] / max(elapsed, 0.1):.1f}")

    # ── Metric row 2 — Intelligence signals ──────────────────────────────
    st.markdown("#### 📡 Intelligence Signal Summary")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Camera Reliability", f"{s['reliability_pct']:.0f}%", help=rel_status)
    r2.metric("Camera Profile", s["camera_id"])
    r3.metric(
        "Coverage Zone",
        "Upload Primary Zone" if "UPLOAD" in s["camera_id"] else "Sector Zones Active",
    )
    r4.metric(
        "Reliability Status",
        rel_status,
    )

    # ── Detected classes breakdown ────────────────────────────────────────
    st.markdown("#### 🔍 Detected Object Classes")
    class_counts = s.get("class_counts", {})
    if class_counts:
        cc_cols = st.columns(min(len(class_counts), 5))
        for i, (cname, cnt) in enumerate(sorted(class_counts.items(), key=lambda x: -x[1])):
            cc_cols[i % len(cc_cols)].metric(f"🏷️ {cname.capitalize()}", f"{cnt} detections")
    else:
        st.info(
            "No objects were classified during this run. "
            "This can happen if the YOLO confidence threshold is too high for the lighting conditions, "
            "or if the video contains only background with no persons, cars, or vehicles visible at ≥50% confidence."
        )

    # ── Reliability note ──────────────────────────────────────────────────
    if s["reliability_pct"] < settings.RELIABILITY_DEGRADED_THRESHOLD:
        st.error(
            f"🔴 **Camera feed quality was POOR ({s['reliability_pct']:.0f}%)** during this analysis. "
            "Detection accuracy is reduced. Evidence from this session should be verified with a secondary camera."
        )
    elif s["reliability_pct"] < settings.RELIABILITY_GOOD_THRESHOLD:
        st.warning(
            f"🟡 **Camera feed quality was DEGRADED ({s['reliability_pct']:.0f}%)** during this analysis. "
            "Actionability is automatically downgraded for alerts from this session."
        )

    # ── Navigation buttons ────────────────────────────────────────────────
    st.markdown("#### Navigate Results")
    nav1, nav2, nav3, nav4 = st.columns(4)
    with nav1:
        if st.button("🚨 VIEW ALERTS", use_container_width=True):
            st.switch_page("pages/alerts.py")
    with nav2:
        if st.button("🔒 VIEW EVIDENCE", use_container_width=True):
            st.switch_page("pages/evidence.py")
    with nav3:
        if st.button("📡 VIEW CAMERA HEALTH", use_container_width=True):
            st.switch_page("pages/camera_health.py")
    with nav4:
        if st.button("📋 VIEW FULL REPORT", use_container_width=True):
            st.switch_page("pages/overview.py")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR STATUS PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Pipeline Status")
st.sidebar.info(
    "**Detection:** YOLOv8n (CPU)\n\n"
    "**Tracking:** ByteTrack / IoU Fallback\n\n"
    f"**Processing:** {settings.PROCESS_FPS:.0f} FPS target\n\n"
    "**Evidence:** SHA-256 tamper-evident\n\n"
    "**Priority:** Independent of Reliability"
)

if demo_degraded:
    st.sidebar.warning("⚡ DEGRADATION DEMO: ON")
    st.sidebar.caption(
        "Programmatic blur (kernel=51) + darkness (0.3x) applied. "
        "Camera Reliability Engine will flag this as POOR."
    )

    # Reset state
    st.session_state["ibvapx_stop_requested"] = False
    st.session_state["ibvapx_analysis_done"] = False
    st.session_state["ibvapx_summary"] = None

    # ── Source / pipeline initialisation ─────────────────────────────────────
    source = None
    pipeline = None
    init_error = None

    try:
        source = FileVideoSource(file_path=selected_file_path, camera_id=camera_id)
        pipeline = IBVAPXPipeline(enable_demo_degradation=demo_degraded)
    except Exception as e:
        init_error = str(e)
        logger.error("Pipeline init error: %s", e)

    if init_error:
        st.error(f"Could not initialise the analysis pipeline. {init_error}")
    else:
        # ── UI placeholders ────────────────────────────────────────────────
        st.markdown(f"#### 📡 Analysis Feed  —  `{source_label}`  ·  Camera: `{camera_id}`")

        if demo_degraded:
            st.warning(
                "🎛️ **Camera Degradation Demo is ON** — "
                "Blur and darkness applied programmatically to simulate a dirty lens."
            )

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        stats_cols = st.columns(6)
        stat_frames = stats_cols[0].empty()
        stat_fps    = stats_cols[1].empty()
        stat_tracks = stats_cols[2].empty()
        stat_events = stats_cols[3].empty()
        stat_hi_pri = stats_cols[4].empty()
        stat_rel    = stats_cols[5].empty()

        # ── Frame loop variables ──────────────────────────────────────────
        total_frames = source.total_frames
        process_interval = max(1, int(round(source.fps / settings.PROCESS_FPS))) if source.fps > 0 else 1
        frame_idx = 0
        processed_count = 0
        total_events = 0
        high_priority_count = 0
        last_reliability_pct = 100.0
        last_tracked_count = 0
        loop_start = time.time()

        try:
            while source.is_connected:
                # ── Stop button check ────────────────────────────────────
                if st.session_state.get("ibvapx_stop_requested", False):
                    st.warning("⏹ Analysis stopped by user. Partial results preserved.")
                    break

                frame_obj = source.get_frame()
                if frame_obj is None:
                    break

                frame_idx += 1

                # ── Frame-skip logic (process every Nth frame only) ──────
                if (frame_idx - 1) % process_interval != 0:
                    continue

                # ── Decode raw numpy image from Frame ───────────────────
                nparr = np.frombuffer(frame_obj.frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # ── Optional resize for inference (keep original for evidence) ──
                infer_img, was_resized = resize_for_inference(img)

                # ── Full IBVAP-X pipeline ─────────────────────────────────
                try:
                    annotated, new_alerts = pipeline.process_frame(source, frame_obj, infer_img)

                    # If we resized for inference, upscale annotation back to display size
                    if was_resized:
                        annotated = cv2.resize(
                            annotated,
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                except Exception as pipe_err:
                    logger.error("Pipeline frame error: %s", pipe_err)
                    annotated = img   # Show raw frame on pipeline error

                new_alerts = new_alerts or []
                processed_count += 1
                total_events += len(new_alerts)
                high_priority_count += sum(
                    1 for a in new_alerts
                    if hasattr(a, "event_priority")
                    and str(a.event_priority).upper() in ("HIGH", "CRITICAL")
                )

                # ── Reliability from pipeline ─────────────────────────────
                try:
                    rel_score = pipeline.reliability_engine.calculate_reliability(
                        camera_id=camera_id,
                        frame_id=frame_obj.frame_id,
                        timestamp=frame_obj.timestamp,
                        image_np=infer_img,
                    )
                    last_reliability_pct = rel_score.composite_score
                except Exception:
                    pass   # Non-blocking

                # ── Track count from last frame ───────────────────────────
                try:
                    last_tracked_count = len(
                        pipeline.tracker.update([], timestamp=frame_obj.timestamp)
                    )
                except Exception:
                    pass

                # ── Annotated frame overlay (source label + camera ID) ──
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 28), (20, 20, 20), -1)
                cv2.putText(
                    annotated,
                    f"[{source_label}]  Cam: {camera_id}  Frame: {frame_idx}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )

                # ── Display annotated frame (BGR → RGB) ──────────────────
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    rgb,
                    caption=f"Frame {frame_idx}/{total_frames} — {source_label} | {camera_id}",
                    use_container_width=True,
                )

                # ── Progress bar ──────────────────────────────────────────
                progress = min(frame_idx / max(total_frames, 1), 1.0)
                progress_bar.progress(progress)

                # ── Live stat metrics ─────────────────────────────────────
                elapsed = time.time() - loop_start
                live_fps = processed_count / elapsed if elapsed > 0 else 0.0
                stat_frames.metric("Frames", f"{frame_idx}/{total_frames}")
                stat_fps.metric("Processing FPS", f"{live_fps:.1f}")
                stat_tracks.metric("Tracked Objects", last_tracked_count)
                stat_events.metric("Events Detected", total_events)
                stat_hi_pri.metric("High/Critical", high_priority_count)
                rel_label = (
                    "GOOD" if last_reliability_pct >= settings.RELIABILITY_GOOD_THRESHOLD
                    else "DEGRADED" if last_reliability_pct >= settings.RELIABILITY_DEGRADED_THRESHOLD
                    else "POOR"
                )
                stat_rel.metric(
                    "Camera Reliability",
                    f"{last_reliability_pct:.0f}% [{rel_label}]",
                )

                # ── Display FPS throttle ──────────────────────────────────
                time.sleep(1.0 / settings.DISPLAY_FPS)

        except Exception as loop_err:
            logger.error("Analysis loop error: %s", loop_err)
            st.error(
                f"An error occurred during analysis. "
                "Partial results have been preserved. "
                f"Error type: {type(loop_err).__name__}"
            )
        finally:
            if source:
                source.release()

        # ── Analysis complete — progress to 100% ──────────────────────────
        if not st.session_state.get("ibvapx_stop_requested", False):
            progress_bar.progress(1.0)
            st.success("✅ Analysis complete.")

        # ── Build summary and save to session state ───────────────────────
        elapsed_total = time.time() - loop_start
        summary = {
            "source_label": source_label,
            "camera_id": camera_id,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "total_events": total_events,
            "high_priority_count": high_priority_count,
            "elapsed_seconds": elapsed_total,
            "reliability_pct": last_reliability_pct,
            "stopped_early": st.session_state.get("ibvapx_stop_requested", False),
        }
        st.session_state["ibvapx_summary"] = summary
        st.session_state["ibvapx_analysis_done"] = True

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING SUMMARY UI
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("ibvapx_analysis_done") and st.session_state.get("ibvapx_summary"):
    s = st.session_state["ibvapx_summary"]
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary")

    tag = "⚠️ STOPPED EARLY" if s["stopped_early"] else "✅ COMPLETE"
    st.markdown(
        f"**Status:** {tag} &nbsp;|&nbsp; "
        f"**Source:** `{s['source_label']}` &nbsp;|&nbsp; "
        f"**Camera:** `{s['camera_id']}`"
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Frames Analysed", f"{s['processed_frames']}/{s['total_frames']}")
    m2.metric("Total Events", s["total_events"])
    m3.metric("High/Critical", s["high_priority_count"])
    m4.metric("Reliability", f"{s['reliability_pct']:.0f}%")
    elapsed = s["elapsed_seconds"]
    m5.metric(
        "Analysis Time",
        f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
    )
    m6.metric(
        "Effective FPS",
        f"{s['processed_frames'] / max(elapsed, 0.1):.1f}",
    )

    st.markdown("#### Navigate Results")
    nav1, nav2, nav3, nav4 = st.columns(4)

    with nav1:
        if st.button("🚨 VIEW ALERTS", use_container_width=True):
            st.switch_page("pages/alerts.py")
    with nav2:
        if st.button("🔒 VIEW EVIDENCE", use_container_width=True):
            st.switch_page("pages/evidence.py")
    with nav3:
        if st.button("📡 VIEW CAMERA HEALTH", use_container_width=True):
            st.switch_page("pages/camera_health.py")
    with nav4:
        if st.button("📋 VIEW FULL REPORT", use_container_width=True):
            st.switch_page("pages/overview.py")

# ─────────────────────────────────────────────────────────────────────────────
# STATIC STATUS PANEL (right sidebar info when no analysis is running)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Pipeline Status")
st.sidebar.info(
    "**Detection:** YOLOv8n (CPU)\n\n"
    "**Tracking:** ByteTrack / IoU Fallback\n\n"
    f"**Processing:** {settings.PROCESS_FPS:.0f} FPS target\n\n"
    "**Evidence:** SHA-256 tamper-evident\n\n"
    "**Priority:** Independent of Reliability"
)

if demo_degraded:
    st.sidebar.warning("⚡ DEGRADATION DEMO: ON")
    st.sidebar.caption(
        "Programmatic blur (kernel=51) + darkness (0.3x) applied. "
        "Camera Reliability Engine will flag this as POOR."
    )
