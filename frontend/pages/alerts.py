import streamlit as st
import time

st.set_page_config(page_title="Alerts — IBVAP-X", page_icon="🚨", layout="wide")

st.title("🚨 Operational Priority Alerts")
st.caption("Reliability-Aware Alert Queue & Decision Explainability Panel")

st.markdown("---")

# Demo mock alerts list for visual demonstration
demo_alerts = [
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

for alert in demo_alerts:
    with st.expansions(f"🚨 [{alert['event_priority']}] Alert #{alert['alert_id']} — Camera {alert['camera_id']} ({alert['time_str']})" if hasattr(st, "expansions") else st.expander(f"🚨 [{alert['event_priority']}] Alert #{alert['alert_id']} — Camera {alert['camera_id']} ({alert['time_str']})", expanded=True)):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_color = "red" if alert['event_priority'] == "HIGH" else "orange"
            st.markdown(f"**Event Priority:** :{p_color}[{alert['event_priority']} ({alert['event_priority_score']:.0f}/100)]")
        with col2:
            r_color = "orange" if alert['camera_reliability'] == "DEGRADED" else "green"
            st.markdown(f"**Camera Reliability:** :{r_color}[{alert['camera_reliability']} ({alert['camera_reliability_score']:.0f}%)]")
        with col3:
            st.markdown(f"**Actionability Rating:** **{alert['actionability']}**")

        st.markdown("#### 📋 Explainability Breakdown (Why Suspicious?)")
        for reason in alert['why_reasons']:
            st.markdown(f"- ✓ {reason}")

        if alert['camera_reliability'] == "DEGRADED":
            st.warning(
                f"⚠️ **Observation Reliability Degraded ({alert['camera_reliability_score']:.0f}%):** "
                f"Detection confidence is high, but camera reliability is degraded. Recommend verifying camera state before escalation."
            )

        st.info(f"💡 **Recommended Action:** {alert['action_recommendation']}")

        # Verification actions
        c_ack, c_rej, c_unc, c_timer = st.columns([1, 1, 1, 2])
        with c_ack:
            if st.button("✅ CONFIRM", key=f"ack_{alert['alert_id']}"):
                st.success("Alert ACKNOWLEDGED by operator.")
        with c_rej:
            if st.button("❌ REJECT", key=f"rej_{alert['alert_id']}"):
                st.info("Alert REJECTED by operator.")
        with c_unc:
            if st.button("❓ UNCERTAIN", key=f"unc_{alert['alert_id']}"):
                st.warning("Alert marked UNCERTAIN.")
        with c_timer:
            st.markdown("⏱️ Escalation Timer: **00:22** (Timeout: 30s)")
