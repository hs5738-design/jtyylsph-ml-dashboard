# =========================================================
# JTYYLSPH V6.2 PRO MAX — ENTERPRISE AI PLATFORM
# Production • Persistent Models • Registry • Explainability
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# =========================================================
# CONFIG
# =========================================================

MODEL_REGISTRY = "model_registry.json"
MODEL_DIR = "models"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# REGISTRY FUNCTIONS
# =========================================================

def load_registry():
    if not os.path.exists(MODEL_REGISTRY):
        return []
    try:
        with open(MODEL_REGISTRY, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except:
        pass
    return []

def save_registry(reg):
    with open(MODEL_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)
def register_model(name, model, feature_names, metrics):
    """Save model artifact and record to registry safely."""
    registry = load_registry()

    # Extract versions safely, ignore invalid values
    versions = []
    for r in registry:
        if r.get("name") == name:
            try:
                v = int(r.get("version", 0))
                versions.append(v)
            except:
                continue

    version = max(versions) + 1 if versions else 1

    model_path = os.path.join(MODEL_DIR, f"{name}_v{version}.pkl")

    # Ensure all metrics exist
    safe_metrics = {
        "accuracy": metrics.get("accuracy", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0)
    }

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "metrics": safe_metrics
    }

    joblib.dump(artifact, model_path)

    record = {
        "name": name,
        "version": version,
        "path": model_path,
        "metrics": safe_metrics,
        "time": datetime.datetime.utcnow().isoformat()
    }

    registry.append(record)
    save_registry(registry)

def load_models_from_registry():
    """Load all models safely from registry."""
    registry = load_registry()
    models = {}
    for rec in registry:
        try:
            path = rec.get("path")
            if not path or not os.path.exists(path):
                continue
            artifact = joblib.load(path)
            model = artifact.get("model")
            if model is None:
                continue
            models[rec["name"]] = artifact
        except:
            continue
    return models

# =========================================================
# HELPERS
# =========================================================

def safe_barh(names, values, title):
    names = list(names)
    values = list(values)
    if len(names) != len(values):
        min_len = min(len(names), len(values))
        names = names[:min_len]
        values = values[:min_len]
    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_title(title)
    st.pyplot(fig)

def align_features(df, feature_order):
    return df.reindex(columns=feature_order, fill_value=0)

def log_prediction(data, pred, prob, model_name):
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "input": data,
        "prediction": int(pred),
        "probability": float(prob) if prob is not None else None,
        "model": model_name
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# =========================================================
# SESSION STATE
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "training_done" not in st.session_state:
    st.session_state.training_done = False
if "feature_names" not in st.session_state:
    st.session_state.feature_names = []

# =========================================================
# AUTO-LOAD MODELS
# =========================================================

if not st.session_state.trained_models:
    loaded = load_models_from_registry()
    if loaded:
        for name, artifact in loaded.items():
            model = artifact.get("model")
            if model is None:
                continue
            st.session_state.trained_models[name] = model
            st.session_state.feature_names = artifact.get("feature_names", st.session_state.feature_names)
            metrics = artifact.get("metrics", {})
            st.session_state.leaderboard[name] = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0)
            }
        st.session_state.training_done = bool(st.session_state.trained_models)

# =========================================================
# UI
# =========================================================

st.title("🚀 JTYYLSPH V6.2 PRO MAX AI PLATFORM")
st.caption("AutoML • MLOps • Registry • Drift Detection • Explainability")

# =========================================================
# DATA
# =========================================================

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
domain = st.sidebar.selectbox("Synthetic Dataset", ["Finance", "Healthcare", "Sports", "General"])

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    X_data, y_data = make_classification(n_samples=500, n_features=6, random_state=42)
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

X.columns = [str(c) for c in X.columns]
feature_names = list(X.columns)
st.session_state.feature_names = feature_names
st.write("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# MODELS
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
# TABS
# =========================================================

tabs = st.tabs(["Training", "Comparison", "Prediction", "Monitoring", "Explainability", "Registry", "Audit Logs"])

# =========================================================
# TRAINING
# =========================================================

with tabs[0]:
    if st.button("Train Models"):
        st.session_state.leaderboard = {}
        for name, model in models.items():
            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted"),
                "recall": recall_score(y_test, preds, average="weighted"),
                "f1": f1_score(y_test, preds, average="weighted")
            }

            st.session_state.trained_models[name] = model
            st.session_state.leaderboard[name] = metrics
            register_model(name, model, feature_names, metrics)

        st.session_state.training_done = True
        st.success("✅ Training Complete")

# =========================================================
# COMPARISON
# =========================================================

with tabs[1]:
    if st.session_state.training_done:
        df_lb = pd.DataFrame(st.session_state.leaderboard).T
        for col in ["accuracy", "precision", "recall", "f1"]:
            if col not in df_lb.columns:
                df_lb[col] = 0.0
        st.dataframe(df_lb)
        st.bar_chart(df_lb["accuracy"])
        champion = df_lb["accuracy"].idxmax()
        st.success(f"Champion Model: {champion}")
    else:
        st.info("Train models first")

# =========================================================
# PREDICTION
# =========================================================

with tabs[2]:
    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        inputs = [st.number_input(f, value=0.0) for f in feature_names]
        model_name = st.selectbox("Model", list(st.session_state.trained_models.keys()))
        if st.button("Predict"):
            model = st.session_state.trained_models[model_name]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            input_df = align_features(input_df, feature_names)
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {pred}")
            log_prediction(input_df.to_dict(orient="records")[0], pred, prob, model_name)

# =========================================================
# MONITORING
# =========================================================

with tabs[3]:
    st.write(X.describe())
    col = st.selectbox("Feature", feature_names)
    stat, p = ks_2samp(X_train[col], X_test[col])
    if p < 0.05:
        st.warning("⚠️ Possible Drift Detected")
    else:
        st.success("No Drift")

# =========================================================
# EXPLAINABILITY
# =========================================================

with tabs[4]:
    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        model_name = st.selectbox("Model", list(st.session_state.trained_models.keys()))
        model = st.session_state.trained_models[model_name]
        if hasattr(model, "feature_importances_"):
            safe_barh(feature_names, model.feature_importances_, "Feature Importance")
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            safe_barh(feature_names, coefs, "Coefficient Importance")
        if SHAP_AVAILABLE and st.button("Run SHAP"):
            try:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test[:100])
                fig = plt.figure()
                shap.summary_plot(shap_values, X_test[:100], show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP error: {e}")

# =========================================================
# REGISTRY
# =========================================================

with tabs[5]:
    reg = load_registry()
    if reg:
        st.dataframe(pd.DataFrame(reg))
    else:
        st.info("No models registered")

# =========================================================
# LOGS
# =========================================================

with tabs[6]:
    if os.path.exists(LOG_FILE):
        logs = [json.loads(l) for l in open(LOG_FILE)]
        st.dataframe(pd.DataFrame(logs))
    else:
        st.info("No logs yet")
