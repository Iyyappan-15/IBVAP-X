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
    st.metric("Database Mode", "AUTO", delta="Active: SQLite Fallback")

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📷 Camera Status & Health Grid")
    
    cam_cols = st.columns(3)
    cams = [
        {"id": "CAM-01", "name": "North Gate Alpha", "status": "GOOD", "rel": 95.0, "location": "Perimeter Fence"},
        {"id": "CAM-02", "name": "Ridge Watchtower", "status": "GOOD", "rel": 98.0, "location": "Elevated Post"},
        {"id": "CAM-03", "name": "Riverine Outpost", "status": "DEGRADED", "rel": 42.0, "location": "Marker 12"},
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
    st.subheader("⚡ Quick Control Panel")
    st.button("🔄 Refresh System Metrics")
    st.button("🚨 Acknowledge All Alerts")
    st.button("⚙️ Open System Settings")
