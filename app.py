# =========================================================
# JTYYLSPH V5 ENTERPRISE AI PLATFORM
# Production-Ready • Persistent Session State • JSON Registry • Audit Logging
# =========================================================

import streamlit as st
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
# CONFIGURATION
# =========================================================

MODEL_REGISTRY = "model_registry.json"
LOG_FILE = "prediction_logs.jsonl"
ARTIFACT_DIR = "artifacts"
REGISTRY_SCHEMA_VERSION = "2.0"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []
if "training_done" not in st.session_state:
    st.session_state.training_done = False

# =========================================================
# TITLE
# =========================================================

st.title("🚀 JTYYLSPH V5 ENTERPRISE AI PLATFORM")
st.caption("AutoML • MLOps • Drift Detection • Explainability • Audit Logging")

# =========================================================
# DATASET UPLOAD / SYNTHETIC DATA
# =========================================================

st.sidebar.header("📂 Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
domain = st.sidebar.selectbox("Synthetic Dataset", ["Finance", "Healthcare", "Sports", "General"])

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

X.columns = [str(c) for c in X.columns]
feature_names = list(X.columns)
st.write("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# MODEL CONFIG
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

param_grids = {
    "RandomForest": {"n_estimators": [100, 200]},
    "GradientBoosting": {"n_estimators": [100, 200]}
}

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def log_prediction(data, pred, prob, model_name):
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "input": data,
        "prediction": int(pred),
        "probability": float(prob) if prob is not None else None,
        "model": model_name
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_model_version(model, name, metrics):
    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)
    else:
        registry = []

    version_tag = f"v{len(registry)+1}"
    model_path = f"{ARTIFACT_DIR}/{name}_{version_tag}.pkl"
    with open(model_path, "wb") as mf:
        pickle.dump(model, mf)

    record = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "version": version_tag,
        "name": name,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "metrics": metrics,
        "artifact_path": model_path
    }
    registry.append(record)
    with open(MODEL_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)

# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "🤖 Training",
    "📊 Model Comparison",
    "🔮 Prediction",
    "📈 Monitoring",
    "⚖️ Fairness",
    "🗂 Registry",
    "📜 Audit Logs"
])

# =========================================================
# TRAINING TAB
# =========================================================

with tabs[0]:
    st.header("AutoML Training")
    if st.button("Train All Models"):
        st.session_state.leaderboard = []
        best_score = 0

        for name, model in models.items():
            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

            st.session_state.trained_models[name] = model
            st.session_state.leaderboard.append({"Model": name, **metrics})
            save_model_version(model, name, metrics)

            if acc > best_score:
                best_score = acc
                st.session_state.best_model = model

        st.session_state.training_done = True
        st.success("✅ Training Complete")

# =========================================================
# MODEL COMPARISON TAB
# =========================================================

with tabs[1]:
    st.header("Model Performance Comparison")
    if st.session_state.training_done:
        df_lb = pd.DataFrame(st.session_state.leaderboard)
        st.dataframe(df_lb)
        fig, ax = plt.subplots()
        ax.bar(df_lb["Model"], df_lb["accuracy"], color="skyblue")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison")
        st.pyplot(fig)
    else:
        st.info("⚠️ Train models first to see comparison charts.")

# =========================================================
# PREDICTION TAB
# =========================================================

with tabs[2]:
    st.header("Manual Prediction")
    if not st.session_state.training_done:
        st.info("⚠️ Train models first to enable prediction.")
    else:
        inputs = []
        for f in feature_names:
            if np.issubdtype(X[f].dtype, np.number):
                val = st.number_input(label=f, value=float(X[f].mean()))
            else:
                val = st.text_input(label=f, value="")
            inputs.append(val)

        model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
        if st.button("Predict"):
            model = st.session_state.trained_models[model_name]
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

with tabs[3]:
    st.header("Data Monitoring & Drift Detection")
    if st.session_state.training_done:
        st.write(X.describe())
        corr = X.corr()
        fig, ax = plt.subplots()
        cax = ax.matshow(corr)
        plt.colorbar(cax)
        st.pyplot(fig)

        st.subheader("Drift Detection")
        col = st.selectbox("Feature for Drift Test", feature_names)
        stat, p = ks_2samp(X_train[col], X_test[col])
        if p < 0.05:
            st.warning("⚠️ Possible Drift Detected")
        else:
            st.success("No significant drift")
    else:
        st.info("⚠️ Train models first to enable monitoring.")

# =========================================================
# REGISTRY TAB
# =========================================================

with tabs[5]:
    st.header("Model Registry")
    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)
        st.dataframe(pd.DataFrame(registry))
    else:
        st.info("No registry entries yet.")

# =========================================================
# AUDIT LOG TAB
# =========================================================

with tabs[6]:
    st.header("Prediction Audit Logs")
    if os.path.exists(LOG_FILE):
        logs = [json.loads(line) for line in open(LOG_FILE)]
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs)

        fig, ax = plt.subplots()
        df_logs["prediction"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution")
        st.pyplot(fig)
    else:
        st.info("No logs yet.")
# =========================================================
# EXPLAINABILITY TAB (V5 ENTERPRISE)
# =========================================================

with tabs[4]:  # Assuming tabs[4] is the "🧠 Explainability" tab
    st.header("Model Explainability")

    if not st.session_state.get("training_done", False):
        st.info("⚠️ Train models first to use explainability features.")
    else:
        model_name = st.selectbox(
            "Select Model for Explainability",
            list(st.session_state.trained_models.keys()),
            key="explain_model"
        )
        model = st.session_state.trained_models[model_name]

        st.subheader("SHAP Explainability")
        if SHAP_AVAILABLE:
            if st.button("Run SHAP", key="shap_button"):
                sample_X = X_test.sample(min(100, len(X_test)), random_state=42)
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(sample_X)
                fig = plt.figure()
                shap.summary_plot(shap_values, sample_X, show=False)
                st.pyplot(fig)
        else:
            st.warning("SHAP not installed. You can still see feature importance below.")

        st.subheader("Feature Importance / Coefficients")
        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) != len(feature_names):
                st.warning(f"Cannot plot feature importances: {len(importances)} != {len(feature_names)}")
            else:
                fig, ax = plt.subplots()
                ax.barh(feature_names, importances, color="skyblue")
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature Importance — {model_name}")
                st.pyplot(fig)
        # Linear models
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            coefs = np.atleast_1d(coefs)
            if len(coefs) != len(feature_names):
                st.warning(f"Cannot plot coefficients: {len(coefs)} != {len(feature_names)}")
            else:
                fig, ax = plt.subplots()
                ax.barh(feature_names, coefs, color="orange")
                ax.set_xlabel("Coefficient Magnitude")
                ax.set_title(f"Feature Coefficients — {model_name}")
                st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model type.")
