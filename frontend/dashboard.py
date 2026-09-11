import streamlit as st

st.set_page_config(
    page_title="IBVAP-X — Border Video Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ IBVAP-X")
st.caption("Reliability-Aware Border Video Intelligence | Smart India Hackathon PS ID: 26187")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Active Cameras", value="3", delta="All Online")
with col2:
    st.metric(label="Camera Reliability", value="94%", delta="GOOD")
with col3:
    st.metric(label="Active Alerts", value="2", delta="1 High Priority")
with col4:
    st.metric(label="System Status", value="OPERATIONAL", delta="PG/SQLite Auto")

st.info("💡 Sprint 1 Skeleton Loaded. Select pages from the sidebar navigation once features are activated.")
