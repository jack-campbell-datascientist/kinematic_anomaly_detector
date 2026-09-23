# Kinematic Anomaly Detector

**Rapid prototype for spatial intelligence / mission-focused data science**

Interactive Streamlit application that detects unusual kinematic behaviors (speed, acceleration, heading changes, path tortuosity) in vehicle trajectories using classical machine learning.

Built to demonstrate the core skills required for roles involving statistical & ML models on live/historical geospatial time-series data, feature engineering, anomaly detection, and analytic visualization.

---

## What it does

1. Ingests multi-object GPS trajectories (T-Drive taxi data)
2. Cleans and engineers kinematic features per trajectory
3. Trains an Isolation Forest anomaly detector
4. Serves an interactive dashboard for exploration, scoring, and interpretation

---

## Project Structure
vantor_kinematic_anomaly_detector/
├── app.py                     # Streamlit application
├── analysis.ipynb             # Data loading, feature engineering, model training
├── requirements.txt
├── README.md
├── data/
│   ├── taxi_log/              # Raw T-Drive .txt files (one per taxi)
│   └── trajectory_features.csv
├── models/
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   └── feature_cols.joblib
└── venv/
text---

## Quick Start

### 1. Setup environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt