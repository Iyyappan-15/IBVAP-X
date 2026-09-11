import streamlit as st
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="Coverage Map — IBVAP-X", page_icon="🗺️", layout="wide")

st.title("🗺️ Camera Coverage & Vulnerability Map")
st.caption("Geometric FOV Wedge Modeling & Blind-Spot Identification")

# MANDATORY DISCLAIMER BANNER (Refinement 4 & 5)
st.warning(
    "⚠️ **GEOMETRIC CAMERA-COVERAGE APPROXIMATION — Demo Simulation Data**\n\n"
    "This map displays simplified geometric FOV estimates calculated in a local Euclidean metric plane. "
    "Modeled FOV coverage is not equivalent to physical surveillance certainty. "
    "Terrain, elevation, weather, camera resolution, and physical obstructions are not accounted for in this prototype model."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("🟢 **2+ modeled camera FOVs:** High Overlapping Coverage")
with col2:
    st.warning("🟡 **1 modeled camera FOV:** Single-Sight Line Coverage")
with col3:
    st.error("🔴 **0 modeled camera FOVs:** Identified Coverage Gap / Blind Spot")

st.markdown("---")

# Build Folium map with demo border camera markers and FOV wedges
map_center = [31.6235, 74.8785]
m = folium.Map(location=map_center, zoom_start=14, tiles="OpenStreetMap")

demo_cams = [
    {"id": "CAM-01", "name": "North Gate Alpha", "lat": 31.6215, "lon": 74.8752, "heading": 45, "fov": 65, "range": 150},
    {"id": "CAM-02", "name": "Central Ridge Watchtower", "lat": 31.6230, "lon": 74.8780, "heading": 90, "fov": 70, "range": 180},
    {"id": "CAM-03", "name": "East Riverine Outpost", "lat": 31.6260, "lon": 74.8820, "heading": 135, "fov": 60, "range": 140},
]

for cam in demo_cams:
    folium.Marker(
        location=[cam["lat"], cam["lon"]],
        popup=f"<b>{cam['id']}</b><br>{cam['name']}<br>Heading: {cam['heading']}°<br>FOV: {cam['fov']}°<br>Range: {cam['range']}m",
        tooltip=cam["id"],
        icon=folium.Icon(color="blue", icon="video", prefix="fa")
    ).add_to(m)

# Highlight identified Blind Spot polygon area between CAM-02 and CAM-03
blindspot_pts = [
    [31.6245, 74.8800],
    [31.6255, 74.8805],
    [31.6250, 74.8815],
    [31.6240, 74.8810]
]
folium.Polygon(
    locations=blindspot_pts,
    color="red",
    weight=3,
    fill=True,
    fill_color="red",
    fill_opacity=0.4,
    popup="<b>IDENTIFIED COVERAGE GAP / BLIND SPOT</b><br>0 modeled camera FOVs in this zone"
).add_to(m)

# Render Folium map in Streamlit
folium_static(m, width=1100, height=550)
