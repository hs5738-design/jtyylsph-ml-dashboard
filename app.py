# =========================================================
# AI Risk Stability Monitor — MVP VERSION
# Clean • Deployable • Business Ready
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wasserstein_distance

# =========================================================
# UI TITLE
# =========================================================
st.title("🧠 AI Risk Stability Monitor")
st.write("Detect model drift, fairness risk, and system stability in real time.")

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("### Preview")
    st.dataframe(df.head())

    # =========================================================
    # TARGET SELECTION
    # =========================================================
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Only numeric
    X = X.select_dtypes(include=[np.number]).fillna(0)

    if len(X.columns) == 0:
        st.error("No numeric features found.")
        st.stop()

    # =========================================================
    # TRAIN TEST SPLIT
    # =========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================================================
    # MODEL TRAINING
    # =========================================================
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # =========================================================
    # METRICS
    # =========================================================

    # Drift (simple)
    drift = wasserstein_distance(
        X_train.iloc[:, 0],
        X_test.iloc[:, 0]
    )

    # Fairness (simple proxy)
    fairness_gap = abs(preds.mean() - y_test.mean())

    # Stability Score
    def system_stability_score(drift, fairness):
        return (
            (1 - drift) * 0.5 +
            (1 - fairness) * 0.5
        )

    stability = system_stability_score(drift, fairness_gap)

    # =========================================================
    # STATUS LABEL
    # =========================================================
    def status_label(value):
        if value < 0.3:
            return "🟢 Stable"
        elif value < 0.6:
            return "🟡 Warning"
        else:
            return "🔴 Critical"

    # =========================================================
    # OUTPUT
    # =========================================================
    st.write("## 📊 Risk Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Drift Score", round(drift, 3), status_label(drift))
    col2.metric("Fairness Gap", round(fairness_gap, 3), status_label(fairness_gap))
    col3.metric("System Stability", round(stability, 3), status_label(1 - stability))

    # =========================================================
    # SIMPLE INTERPRETATION
    # =========================================================
    st.write("## 🧾 Interpretation")

    if drift > 0.3:
        st.warning("Model drift detected — retraining recommended.")

    if fairness_gap > 0.1:
        st.warning("Potential bias detected — review model fairness.")

    if stability < 0.5:
        st.error("System stability is low — immediate action required.")
    else:
        st.success("System operating within stable range.")

else:
    st.info("Upload a CSV file to begin.")
