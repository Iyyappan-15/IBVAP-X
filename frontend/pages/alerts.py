import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import time
from datetime import datetime
from backend.evidence.audit import AuditLogger

st.set_page_config(page_title="Alerts — IBVAP-X", page_icon="🚨", layout="wide")

st.title("🚨 Operational Priority Alerts")
st.caption("Reliability-Aware Alert Queue & Decision Explainability Panel")

st.markdown("---")

# Retrieve live alerts from session state or use default demonstration alerts
session_alerts = []
if st.session_state.get("ibvapx_summary") and st.session_state["ibvapx_summary"].get("alerts_list"):
    for a in st.session_state["ibvapx_summary"]["alerts_list"]:
        session_alerts.append({
            "alert_id": a["alert_id"],
            "camera_id": st.session_state["ibvapx_summary"]["camera_id"],
            "track_id": a.get("track_id", 1),
            "class_name": a.get("class_name", "target"),
            "time_str": datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S"),
            "event_priority": a["priority"],
            "event_priority_score": a["priority_score"],
            "camera_reliability": a["camera_reliability"],
            "camera_reliability_score": a.get("camera_reliability_score", 95.0),
            "actionability": a["actionability"],
            "action_recommendation": a["action_recommendation"],
            "why_reasons": a["why_reasons"],
            "state": st.session_state.get("operator_actions", {}).get(a["alert_id"], {}).get("status", "ACTIVE")
        })

demo_alerts = session_alerts or [
    {
        "alert_id": "ALT-8F92A1",
        "camera_id": "CAM-03",
        "track_id": 17,
        "class_name": "person",
        "time_str": "19:42:12",
        "event_priority": "HIGH",
        "event_priority_score": 78.0,
        "camera_reliability": "DEGRADED",
        "camera_reliability_score": 42.0,
        "actionability": "MEDIUM",
        "action_recommendation": "High priority event observed on DEGRADED camera feed. Verify camera condition before escalation.",
        "why_reasons": [
            "Restricted zone entry detected (Restricted Waterway Delta)",
            "Night-time operational context event (NIGHT)",
            "Sustained loitering detected (43s > threshold)",
            "Trajectory movement vector directed toward border boundary (TOWARD_BOUNDARY)"
        ],
        "state": "ACTIVE"
    },
    {
        "alert_id": "ALT-4C10B2",
        "camera_id": "CAM-01",
        "track_id": 12,
        "class_name": "car",
        "time_str": "19:35:04",
        "event_priority": "MEDIUM",
        "event_priority_score": 50.0,
        "camera_reliability": "GOOD",
        "camera_reliability_score": 94.0,
        "actionability": "MEDIUM",
        "action_recommendation": "Medium operational priority event. Standard operator review queued.",
        "why_reasons": [
            "Night-time operational context event (NIGHT)",
            "Restricted zone entry detected (Sector Alpha)"
        ],
        "state": "ACKNOWLEDGED"
    }
]

# Priority colour helper
PRIORITY_COLOURS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

STATE_BADGES = {
    "ACTIVE":       "🔵 ACTIVE",
    "ACKNOWLEDGED": "✅ ACKNOWLEDGED",
    "REJECTED":     "❌ REJECTED",
    "UNCERTAIN":    "❓ UNCERTAIN",
    "ESCALATED":    "⬆️ ESCALATED",
}

for alert in demo_alerts:
    icon = PRIORITY_COLOURS.get(alert["event_priority"], "⚪")
    state_badge = STATE_BADGES.get(alert["state"], alert["state"])
    header = (
        f"{icon} [{alert['event_priority']}] Alert #{alert['alert_id']} "
        f"— Camera {alert['camera_id']} | {alert['class_name'].upper()} | {alert['time_str']} | {state_badge}"
    )

    with st.expander(header, expanded=(alert["state"] == "ACTIVE")):
        col1, col2, col3 = st.columns(3)

        with col1:
            p_color = (
                "red"    if alert["event_priority"] in ("CRITICAL", "HIGH")
                else "orange" if alert["event_priority"] == "MEDIUM"
                else "green"
            )
            st.markdown(
                f"**Event Priority**  \n"
                f":{p_color}[**{alert['event_priority']}** "
                f"({alert['event_priority_score']:.0f} / 100)]"
            )

        with col2:
            r_color = (
                "green"  if alert["camera_reliability"] == "GOOD"
                else "orange" if alert["camera_reliability"] == "DEGRADED"
                else "red"
            )
            st.markdown(
                f"**Camera Reliability**  \n"
                f":{r_color}[**{alert['camera_reliability']}** "
                f"({alert['camera_reliability_score']:.0f}%)]"
            )

        with col3:
            a_color = (
                "green"  if alert["actionability"] == "HIGH"
                else "orange" if alert["actionability"] == "MEDIUM"
                else "red"
            )
            st.markdown(
                f"**Actionability**  \n"
                f":{a_color}[**{alert['actionability']}**]"
            )

        st.markdown("#### 📋 Explainability Breakdown — Why Alerted?")
        for reason in alert["why_reasons"]:
            st.markdown(f"- ✓ {reason}")

        if alert["camera_reliability"] == "DEGRADED":
            st.warning(
                f"⚠️ **Observation Reliability Degraded ({alert['camera_reliability_score']:.0f}%):** "
                f"Detection confidence is high, but camera reliability is degraded. "
                f"Recommend verifying camera state before escalation."
            )
        elif alert["camera_reliability"] == "POOR":
            st.error(
                f"🔴 **Camera POOR ({alert['camera_reliability_score']:.0f}%):** "
                f"Feed quality is critically low. Actionability is automatically downgraded. "
                f"Deploy physical verification."
            )

        st.info(f"💡 **Recommended Action:** {alert['action_recommendation']}")

        # Operator verification actions
        c_ack, c_rej, c_unc, c_timer = st.columns([1, 1, 1, 2])
        with c_ack:
            if st.button("✅ CONFIRM", key=f"ack_{alert['alert_id']}"):
                AuditLogger().log_event("OPERATOR_ACTION", "OPERATOR-01", alert["alert_id"], "Confirmed alert.")
                st.success("Alert ACKNOWLEDGED by operator.")
        with c_rej:
            if st.button("❌ REJECT", key=f"rej_{alert['alert_id']}"):
                AuditLogger().log_event("OPERATOR_ACTION", "OPERATOR-01", alert["alert_id"], "Rejected alert (false positive).")
                st.info("Alert REJECTED — logged as false positive.")
        with c_unc:
            if st.button("❓ UNCERTAIN", key=f"unc_{alert['alert_id']}"):
                AuditLogger().log_event("OPERATOR_ACTION", "OPERATOR-01", alert["alert_id"], "Marked alert uncertain.")
                st.warning("Alert marked UNCERTAIN — flagged for supervisor review.")
        with c_timer:
            st.markdown("⏱️ Escalation Policy: **Auto-escalates in 30s if unacknowledged**")

st.markdown("---")
st.caption(
    "ℹ️ Alerts are populated dynamically from the Live Monitoring session or available as demonstration records."
)

