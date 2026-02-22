# =========================================================
# JTYYLSPH Intelligent Classification Platform V3
# Enterprise / MLOps Architecture
# =========================================================

import streamlit as st
st.set_page_config(page_title="JTYYLSPH AI Platform V3", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle
import os
import io

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from scipy.stats import ks_2samp

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# =========================================================
# TITLE
# =========================================================

st.title("🚀 Intelligent Classification & Analytics Platform — V3")
st.caption("Enterprise ML • AutoML • Monitoring • Explainability")

# =========================================================
# UTILITIES
# =========================================================

LOG_FILE = "prediction_logs.jsonl"
MODEL_REGISTRY = "model_registry.pkl"

def log_prediction(data, pred, prob, model_name):
    entry = {
        "time": datetime.datetime.now().isoformat(),
        "input": data,
        "prediction": int(pred),
        "probability": float(prob) if prob is not None else None,
        "model": model_name,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_model_version(model, name, metrics):
    record = {
        "name": name,
        "timestamp": datetime.datetime.now(),
        "metrics": metrics,
        "model": model
    }
    if os.path.exists(MODEL_REGISTRY):
        registry = pickle.load(open(MODEL_REGISTRY, "rb"))
    else:
        registry = []
    registry.append(record)
    pickle.dump(registry, open(MODEL_REGISTRY, "wb"))

# =========================================================
# DATASET SIDEBAR
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "General"]
)

# ----------------------------
# Load or Generate Dataset
# ----------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    if domain == "Finance":
        X_data, y_data = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=42)
    elif domain == "Healthcare":
        X_data, y_data = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=1)
    elif domain == "Sports":
        X_data, y_data = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=7)
    else:
        X_data, y_data = make_classification(n_samples=400, n_features=4, random_state=0)
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

# Ensure column names are strings
X.columns = [str(col) for col in X.columns]
feature_names = list(X.columns)

st.write("Dataset Shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# MODELS
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

param_grids = {
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 5]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
}

trained_models = {}
leaderboard = []
best_model = None
best_score = 0

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Training",
    "🔮 Prediction",
    "📊 Monitoring",
    "🧠 Explainability",
    "📁 Model Registry"
])

# =========================================================
# TRAINING TAB
# =========================================================

with tab1:
    st.header("AutoML Training")
    if st.button("Train Models"):
        leaderboard = []
        global best_score, best_model
        best_score = 0
        best_model = None

        for name, model in models.items():
            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            trained_models[name] = model

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")
            cv_scores = cross_val_score(model, X, y, cv=5)

            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cv_mean": cv_scores.mean()}
            leaderboard.append({"Model": name, **metrics})
            save_model_version(model, name, metrics)

            if acc > best_score:
                best_score = acc
                best_model = model

            st.subheader(name)
            st.write(metrics)

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
            st.pyplot(fig_cm)

            # Multi-class ROC
            if hasattr(model, "predict_proba"):
                y_classes = np.unique(y)
                if len(y_classes) == 2:
                    probs = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC {roc_auc:.2f}")
                    ax.plot([0,1],[0,1], linestyle="--")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    y_test_bin = label_binarize(y_test, classes=y_classes)
                    probs = model.predict_proba(X_test)
                    fig, ax = plt.subplots()
                    for i, class_label in enumerate(y_classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {class_label} AUC {roc_auc:.2f}")
                    ax.plot([0,1],[0,1], linestyle="--")
                    ax.legend()
                    st.pyplot(fig)

        st.success("Training Complete")
        st.dataframe(pd.DataFrame(leaderboard))

# =========================================================
# PREDICTION TAB
# =========================================================

with tab2:
    st.header("Manual Prediction")
    if not trained_models:
        st.info("Train models first to enable manual prediction.")
    else:
        inputs = []
        for f in feature_names:
            if np.issubdtype(X[f].dtype, np.number):
                val = st.number_input(label=f, value=float(X[f].mean()))
            else:
                val = st.text_input(label=f, value="")
            inputs.append(val)

        model_name = st.selectbox("Select Model", list(trained_models.keys()))
        if st.button("Predict"):
            model = trained_models[model_name]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            for f in feature_names:
                if np.issubdtype(X[f].dtype, np.number):
                    input_df[f] = pd.to_numeric(input_df[f], errors="coerce")
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {pred}")
            if prob is not None:
                st.metric("Confidence", f"{prob:.2f}")
                st.metric("Risk Score", f"{prob*100:.1f}")
            log_prediction(input_df.to_dict(orient="records")[0], pred, prob, model_name)

# =========================================================
# MONITORING TAB
# =========================================================

with tab3:
    st.header("Data Monitoring & Drift Detection")
    st.write(X.describe())
    corr = X.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.colorbar(cax)
    st.pyplot(fig)
    st.subheader("Drift Detection")
    col = st.selectbox("Feature", feature_names)
    stat, p = ks_2samp(X_train[col], X_test[col])
    if p < 0.05:
        st.warning("⚠️ Possible Drift Detected")
    else:
        st.success("No significant drift")

# =========================================================
# EXPLAINABILITY TAB
# =========================================================

with tab4:
    st.header("Explainability")
    if not SHAP_AVAILABLE:
        st.warning("SHAP not installed.")
    elif not trained_models:
        st.info("Train models first.")
    else:
        model_name = st.selectbox("Model", list(trained_models.keys()), key="shap")
        if st.button("Run SHAP"):
            model = trained_models.get(model_name)
            if model:
                sample_X = X_test.sample(min(100, len(X_test)), random_state=42)
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(sample_X)
                fig = plt.figure()
                shap.summary_plot(shap_values, sample_X, show=False)
                st.pyplot(fig)

# =========================================================
# MODEL REGISTRY TAB
# =========================================================

with tab5:
    st.header("Model Registry")
    if os.path.exists(MODEL_REGISTRY):
        registry = pickle.load(open(MODEL_REGISTRY, "rb"))
        rows = []
        for r in registry:
            rows.append({"name": r["name"], "timestamp": r["timestamp"], **r["metrics"]})
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No models saved yet.")

# =========================================================
# EXPORT BEST MODEL
# =========================================================

st.sidebar.header("Export")
if st.sidebar.button("Download Best Model"):
    if best_model is None:
        st.sidebar.warning("Train models first.")
    else:
        buffer = io.BytesIO()
        pickle.dump(best_model, buffer)
        buffer.seek(0)
        st.sidebar.download_button(
            "Download Best Model",
            buffer,
            file_name="best_model.pkl"
        )
