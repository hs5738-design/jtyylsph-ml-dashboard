# app.py

import streamlit as st
import pandas as pd

from core import simulate_stream
from report import generate_pdf_report
from governance import (
    compute_drift,
    compute_fairness,
    system_stability_score,
    status_label
)

# =============================
# PAGE TITLE
# =============================
st.title("🚀 JTYYLSPH — AI Governance Platform")

# =============================
# SESSION STATE
# =============================
if "model" not in st.session_state:
    st.session_state.model = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

# =============================
# SIDEBAR
# =============================
st.sidebar.header("Compliance Mode")

jurisdiction = st.sidebar.selectbox(
    "Regulatory Framework",
    [
        "US (SR 11-7)",
        "EU AI Act",
        "UK Model Risk",
        "APAC Framework",
        "Custom Policy",
    ],
)

if jurisdiction == "EU AI Act":
    st.warning("⚠ High-risk system — audit required")

st.sidebar.header("Dataset Source")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =============================
# DATA LOADING
# =============================
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded Dataset Preview")
    st.dataframe(df.head())

    target_col = st.sidebar.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

else:
    from sklearn.datasets import make_classification

    X_data, y_data = make_classification(
        n_samples=500, n_features=6, random_state=42
    )

    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

# =============================
# TRAIN TEST SPLIT
# =============================
from sklearn.model_selection import train_test_split

X.columns = [str(c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# DATA QUALITY
# =============================
st.subheader("📊 Data Quality Check")

st.write("Missing Values")
st.write(X.isna().sum())

st.write("Duplicate Rows")
st.write(X.duplicated().sum())

# =============================
# MODEL SELECTION
# =============================
model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "GradientBoosting", "LogisticRegression"]
)

# =============================
# TRAIN
# =============================
if st.button("Train Model"):

    if model_choice == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()

    elif model_choice == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()

    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    drift = compute_drift(X_train, X_test)
    fairness = compute_fairness(preds, y_test)
    stability = system_stability_score(drift, fairness)

    st.session_state.model = model
    st.session_state.metrics = (drift, fairness, stability)

# =============================
# SHOW RESULTS (PERSISTENT)
# =============================
if st.session_state.metrics:

    drift, fairness, stability = st.session_state.metrics

    st.subheader("📊 Risk Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Drift", round(drift, 3), status_label(drift))
    c2.metric("Fairness", round(fairness, 3), status_label(fairness))
    c3.metric("Stability", round(stability, 3), status_label(1 - stability))

    # =============================
    # PDF REPORT
    # =============================
    if st.button("📄 Generate Report"):
        file_path = generate_pdf_report(drift, fairness, stability)

        with open(file_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="AI_Risk_Report.pdf",
                mime="application/pdf"
            )

    # =============================
    # INTERPRETATION
    # =============================
    st.subheader("🧾 Interpretation")

    if drift > 0.3:
        st.warning("Drift detected — retrain model")

    if fairness > 0.1:
        st.warning("Bias risk detected")

    if stability < 0.5:
        st.error("🔴 HIGH RISK SYSTEM")
    else:
        st.success("System stable")

    # =============================
    # REAL-TIME SIMULATION
    # =============================
    st.subheader("🧠 Real-Time Simulation")

    if st.button("Start Simulation"):

        chart = st.line_chart()

        model = st.session_state.model

        for step, current_data in simulate_stream(X_test):

            preds = model.predict(current_data)

            drift = compute_drift(X_train, current_data)
            fairness = compute_fairness(preds, y_test)

            chart.add_rows({
                "Drift": [drift],
                "Fairness": [fairness]
            })
