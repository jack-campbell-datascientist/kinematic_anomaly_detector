import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from branca.element import Figure, Element

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Kinematic Anomaly Detector | Vantor-style Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Load artifacts
# -------------------------------------------------
@st.cache_data
def load_data():
    features_path = Path("data/trajectory_features.csv")
    if not features_path.exists():
        st.error("data/trajectory_features.csv not found. Run the analysis notebook first.")
        st.stop()
    return pd.read_csv(features_path, index_col=0)

@st.cache_data
def load_map_data():
    path = Path("data/map_trajectories.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_resource
def load_model():
    model_path = Path("models/isolation_forest.joblib")
    scaler_path = Path("models/scaler.joblib")
    cols_path = Path("models/feature_cols.joblib")

    if not all(p.exists() for p in [model_path, scaler_path, cols_path]):
        st.error("Model artifacts missing. Run the analysis notebook first.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    return model, scaler, feature_cols

df = load_data()
map_df = load_map_data()
model, scaler, feature_cols = load_model()

# Ensure anomaly columns exist
if "anomaly_score" not in df.columns or "is_anomaly" not in df.columns:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_scaled = scaler.transform(X)
    df.loc[X.index, "anomaly_score"] = -model.decision_function(X_scaled)
    df.loc[X.index, "is_anomaly"] = model.predict(X_scaled) == -1

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🛰️ Controls")
st.sidebar.markdown("**Kinematic Anomaly Detector**")
st.sidebar.caption("Detect unusual movement patterns in geospatial trajectories")

score_threshold = st.sidebar.slider(
    "Anomaly score threshold",
    min_value=float(df["anomaly_score"].min()),
    max_value=float(df["anomaly_score"].max()),
    value=float(df["anomaly_score"].quantile(0.95)),
    step=0.01
)

show_only_anomalies = st.sidebar.checkbox("Show only anomalies", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Features used by model")
st.sidebar.write(feature_cols)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("Kinematic Anomaly Detector")
st.markdown(
    """
    **Spatial intelligence demo** – detect unusual kinematic behaviors 
    (speed, acceleration, heading changes, path tortuosity) in vehicle trajectories.
    """
)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_traj = len(df)
n_anom = int(df["is_anomaly"].sum())
pct_anom = n_anom / total_traj * 100
avg_score = df["anomaly_score"].mean()

col1.metric("Trajectories analyzed", f"{total_traj:,}")
col2.metric("Anomalies detected", f"{n_anom:,}", f"{pct_anom:.1f}%")
col3.metric("Avg anomaly score", f"{avg_score:.3f}")
col4.metric("Model contamination", "5%")

st.markdown("---")

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Trajectory Map",
    "📊 Overview & Scores",
    "🔍 Feature Analysis",
    "🚨 Top Anomalies",
    "ℹ️ Method & Notes"
])

# =================================================
# TAB 1 – Full Trajectory Map
# =================================================
with tab1:
    st.subheader("Full Trajectory Paths")

    if map_df is None:
        st.warning(
            "Map data not found. Please run the extra cell at the end of the analysis notebook "
            "to create `data/map_trajectories.parquet`."
        )
    else:
        # Filter
        if show_only_anomalies:
            plot_df = map_df[map_df["is_anomaly"] == True].copy()
        else:
            plot_df = map_df.copy()

        # Limit number of trajectories shown for performance
        unique_ids = plot_df["taxi_id"].unique()
        max_show = st.slider("Max trajectories to draw on map", 5, min(80, len(unique_ids)), 25)

        if len(unique_ids) > max_show:
            # Prefer anomalies
            anom_ids = plot_df[plot_df["is_anomaly"]]["taxi_id"].unique()
            normal_ids = [i for i in unique_ids if i not in anom_ids]
            selected = list(anom_ids)[:max_show]
            remaining = max_show - len(selected)
            if remaining > 0:
                selected += list(np.random.choice(normal_ids, size=min(remaining, len(normal_ids)), replace=False))
            plot_df = plot_df[plot_df["taxi_id"].isin(selected)]

        st.caption(f"Drawing {plot_df['taxi_id'].nunique()} trajectories ({len(plot_df):,} points)")

        # Center map on the data
        center_lat = plot_df["lat"].mean()
        center_lon = plot_df["lon"].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        # Draw each trajectory
        for taxi_id, group in plot_df.groupby("taxi_id"):
            group = group.sort_values("datetime")
            is_anom = group["is_anomaly"].iloc[0]
            color = "#e74c3c" if is_anom else "#3498db"
            weight = 3.5 if is_anom else 1.8
            opacity = 0.85 if is_anom else 0.45

            points = list(zip(group["lat"], group["lon"]))
            folium.PolyLine(
                points,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=f"Taxi {taxi_id} {'(ANOMALY)' if is_anom else ''}"
            ).add_to(m)

            # Start / end markers for anomalies
            if is_anom:
                folium.CircleMarker(
                    location=points[0],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.9,
                    tooltip="Start"
                ).add_to(m)
                folium.CircleMarker(
                    location=points[-1],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.9,
                    tooltip="End"
                ).add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                    background: white; padding: 10px 14px; border-radius: 6px;
                    border: 1px solid #ccc; font-size: 13px;">
            <b>Legend</b><br>
            <span style="color:#e74c3c;">━━</span> Anomalous trajectory<br>
            <span style="color:#3498db;">━━</span> Normal trajectory
        </div>
        """
        m.get_root().html.add_child(Element(legend_html))

        st_folium(m, width=None, height=650, returned_objects=[])

        st.markdown(
            """
            **How to read the map**  
            - Red paths = trajectories the model flagged as anomalous  
            - Blue paths = normal trajectories  
            - Thicker red lines + start/end markers highlight the unusual kinematic behavior
            """
        )

# =================================================
# TAB 2 – Overview
# =================================================
with tab2:
    st.subheader("Anomaly Score Distribution")

    fig_hist = px.histogram(
        df, x="anomaly_score", color="is_anomaly", nbins=50,
        title="Anomaly Score Distribution (higher = more anomalous)",
        color_discrete_map={True: "#e74c3c", False: "#3498db"}
    )
    fig_hist.add_vline(x=score_threshold, line_dash="dash", line_color="orange")
    st.plotly_chart(fig_hist, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            df, x="mean_speed_kmh", y="anomaly_score", color="is_anomaly",
            title="Mean Speed vs Anomaly Score",
            color_discrete_map={True: "#e74c3c", False: "#3498db"},
            hover_data=["std_speed_kmh", "tortuosity"]
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            df, x="mean_heading_change", y="anomaly_score", color="is_anomaly",
            title="Mean Heading Change vs Anomaly Score",
            color_discrete_map={True: "#e74c3c", False: "#3498db"},
            hover_data=["std_accel", "tortuosity"]
        )
        st.plotly_chart(fig2, use_container_width=True)

# =================================================
# TAB 3 – Feature Analysis
# =================================================
with tab3:
    st.subheader("Feature Distributions by Anomaly Flag")
    feature_to_plot = st.selectbox("Select feature", [c for c in feature_cols if c in df.columns])
    fig_box = px.box(
        df, x="is_anomaly", y=feature_to_plot, color="is_anomaly",
        title=f"{feature_to_plot} by Anomaly Status",
        color_discrete_map={True: "#e74c3c", False: "#3498db"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Correlation with Anomaly Score")
    corr = df[feature_cols + ["anomaly_score"]].corr()["anomaly_score"].drop("anomaly_score").sort_values()
    fig_corr = px.bar(x=corr.values, y=corr.index, orientation="h",
                      title="Feature Correlation with Anomaly Score")
    st.plotly_chart(fig_corr, use_container_width=True)

# =================================================
# TAB 4 – Top Anomalies
# =================================================
with tab4:
    st.subheader("Most Anomalous Trajectories")
    top_n = st.slider("Show top N", 5, 50, 15)
    top_anom = (
        df.sort_values("anomaly_score", ascending=False)
          .head(top_n)[
              ["n_points", "duration_min", "total_dist_km",
               "mean_speed_kmh", "std_speed_kmh",
               "mean_heading_change", "tortuosity", "anomaly_score"]
          ]
    )
    st.dataframe(top_anom.style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
                 use_container_width=True)

    st.info(
        """
        **Interpretation tips**  
        - High mean/std speed + high heading change → aggressive or erratic driving  
        - Very high tortuosity → looping / loitering  
        - Extreme acceleration → sudden stops or harsh maneuvers
        """
    )

# =================================================
# TAB 5 – Method
# =================================================
with tab5:
    st.subheader("Method Overview")
    st.markdown(
        """
        ### Pipeline
        1. **Data** – T-Drive taxi trajectories (Beijing GPS)
        2. **Cleaning** – time sorting, invalid point removal, speed/accel sanity filters
        3. **Feature Engineering** (per trajectory)
           - Speed statistics • Acceleration statistics  
           - Heading change magnitude • Path tortuosity  
           - High-speed fraction
        4. **Model** – Isolation Forest (unsupervised, contamination = 5%)
        5. **Visualization** – interactive scores + full trajectory paths on map

        ### Why this matters
        Demonstrates end-to-end handling of geospatial time-series, kinematic feature engineering, 
        unsupervised anomaly detection, and operational visualization — the exact skills needed 
        for mission-focused spatial intelligence work.
        """
    )

st.markdown("---")
st.caption("Rapid prototype – kinematic behavior detection on geospatial trajectories")