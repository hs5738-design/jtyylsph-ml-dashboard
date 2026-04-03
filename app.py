# app.py

import streamlit as st
import pandas as pd

from core import simulate_stream, prepare_data, train_model, predict
from report import generate_pdf_report
from governance import (
    compute_drift,
    compute_fairness,
    system_stability_score,
    status_label
)

st.title("🧠 AI Risk Stability Monitor")

# =============================
# SESSION STATE (CRITICAL)
# =============================
if "model" not in st.session_state:
    st.session_state.model = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None


# =============================
# FILE UPLOAD
# =============================
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X_train, X_test, y_train, y_test = prepare_data(df, target)

    # =============================
    # TRAIN MODEL
    # =============================
    if st.button("Train Model"):
        model = train_model(X_train, y_train)
        preds = predict(model, X_test)

        drift = compute_drift(X_train, X_test)
        fairness = compute_fairness(preds, y_test)
        stability = system_stability_score(drift, fairness)

        # ✅ SAVE STATE
        st.session_state.model = model
        st.session_state.metrics = (drift, fairness, stability)

    # =============================
    # SHOW METRICS (if trained)
    # =============================
    if st.session_state.metrics:
        drift, fairness, stability = st.session_state.metrics

        st.subheader("📊 Risk Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("Drift", round(drift, 3), status_label(drift))
        c2.metric("Fairness", round(fairness, 3), status_label(fairness))
        c3.metric("Stability", round(stability, 3), status_label(1 - stability))

        # =============================
        # PDF DOWNLOAD
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
            st.error("System unstable")
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

else:
    st.info("Upload a dataset to begin.")
