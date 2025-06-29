import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium

ward_data = pd.read_csv("data/processed/wards_with_predictions.csv")
geojson = gpd.read_file("geo/delhi_wards.geojson")

geojson = geojson.set_index("Ward_Name").join(
    ward_data.set_index("Ward_Name"), how="left", lsuffix="", rsuffix="_csv"
).reset_index()

color_map = {
    "underserved": "#FF4B4B",  # red
    "optimal": "#29CC97",      # green
    "cluttered": "#3A86FF"     # blue
}

st.set_page_config(page_title="Delhi Transport Insights", layout="wide")
st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86AB;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #F0F4F8, #E6ECF2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🚌 Delhi Public Transport Coverage</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Assess public transit accessibility across Delhi wards using ML predictions and urban data.</div><br>', unsafe_allow_html=True)

st.sidebar.header("🎛️ Filter Options")
status_filter = st.sidebar.multiselect(
    "Coverage Status",
    options=["underserved", "optimal", "cluttered"],
    default=["underserved", "optimal", "cluttered"]
)

m = folium.Map(location=[28.61, 77.23], zoom_start=11, tiles="cartodbpositron")

for _, row in geojson.iterrows():
    label = row.get("predicted_label")
    color = color_map.get(label, "gray")

    if label in status_filter or pd.isna(label):
        status_display = f"<span style='color:{color}; font-weight: bold'>{label.title() if pd.notna(label) else 'N/A'}</span>"
        
        tooltip_html = f"""
        <div style="font-size:14px">
            <b>Ward:</b> {row['Ward_Name']}<br>
            <b>Status:</b> {status_display}<br>
            <b>Population Density:</b> {int(row['population_density']) if pd.notna(row.get('population_density')) else 'N/A'}<br>
            <b>Stop Density:</b> {row.get('stop_density', 'N/A')}<br>
            <b>Route Density:</b> {row.get('route_density', 'N/A')}
        </div>
        """

        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, clr=color: {
                "fillColor": clr,
                "color": "#333",
                "weight": 0.5,
                "fillOpacity": 0.5,
            },
            tooltip=folium.Tooltip(tooltip_html, sticky=True)
        ).add_to(m)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Interactive Ward Map")
    st_folium(m, width=950, height=600)

with col2:
    st.subheader("📊 Stats Summary")
    st.metric("Total Wards", len(geojson))
    st.metric("Underserved", (geojson["label"] == "underserved").sum())
    st.metric("Optimal", (geojson["label"] == "optimal").sum())
    st.metric("Cluttered", (geojson["label"] == "cluttered").sum())

st.markdown("---")

st.subheader("📋 Ward-wise Predictions Table")
filtered_table = ward_data[ward_data["label"].isin(status_filter)].reset_index(drop=True)
st.dataframe(filtered_table, use_container_width=True, hide_index=True)
