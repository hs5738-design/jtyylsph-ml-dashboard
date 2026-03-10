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
from scipy.stats import wasserstein_distance, ks_2samp
import hashlib

language = st.sidebar.selectbox(
    "Language",
    ["English", "Spanish", "French", "German", "Mandarin (beta)"]
)
# =========================================================
# CONFIG
# =========================================================
PREDICTION_DRIFT_LOG = "drift_logs.jsonl"
MODEL_REGISTRY = "model_registry.json"
MODEL_DIR = "models"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# REGISTRY FUNCTIONS
# =========================================================

def model_hash(path): 
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

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

def log_drift_metrics(feature_name, train_values, test_values, metric_value, metric_name):
    """Log drift monitoring metrics"""
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "feature": feature_name,
        "metric": metric_name,
        "drift_score": float(metric_value)
    }
    try:
        with open(PREDICTION_DRIFT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass
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
# AUTO-LOAD MODELS FROM REGISTRY
# =========================================================

if not st.session_state.trained_models:

    loaded = load_models_from_registry()

    if loaded:
        for name, artifact in loaded.items():
            # Extract the model safely
            model = artifact.get("model")
            if model is None:
                continue  # skip corrupted entries

            st.session_state.trained_models[name] = model

            # Extract feature names
            features = artifact.get("feature_names")
            if features:
                st.session_state.feature_names = features

            # Extract metrics and provide defaults if missing
            metrics = artifact.get("metrics", {})
            safe_metrics = {
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0)
            }
            st.session_state.leaderboard[name] = safe_metrics

        # Sort leaderboard by accuracy descending
        st.session_state.leaderboard = dict(
            sorted(
                st.session_state.leaderboard.items(),
                key=lambda item: item[1].get("accuracy", 0.0),
                reverse=True
            )
        )

        # Mark training as done if any models loaded
        st.session_state.training_done = bool(st.session_state.trained_models)

# =========================================================
# UI
# =========================================================

st.title("🚀JTYYLSPH — AI Governance & Model Risk Infrastructure")
st.caption("Regulatory Stress Testing • Governance Reporting • Model Risk Monitoring")
st.caption("AutoML • MLOps • Registry • Drift Detection • Explainability")

# =========================================================
# DATA
# =========================================================

st.sidebar.header("Compliance Mode")
jurisdiction = st.sidebar.selectbox(
    "Select Regulatory Framework",
    ["United States (SR 11-7)", 
     "European Union (EU AI Act)", 
     "UK Model Risk Guidance",
     "APAC General Risk Framework",
     "Custom Enterprise Policy"]
)
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

tabs = st.tabs([
    "Training",
    "Governance Report",
    "Bias & Fairness",
    "Stress Testing",
    "Monitoring",
    "Explainability",
    "Registry",
    "Audit Logs"
])

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

from sklearn.metrics import confusion_matrix

def fairness_analysis(model, X_test, y_test, sensitive_feature=None):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    result = {
        "overall_accuracy": acc
    }

    if sensitive_feature and sensitive_feature in X_test.columns:
        groups = X_test[sensitive_feature].unique()
        group_metrics = {}
        for g in groups:
            mask = X_test[sensitive_feature] == g
            if mask.sum() > 0:
                group_acc = accuracy_score(y_test[mask], preds[mask])
                group_metrics[str(g)] = group_acc
        result["group_accuracy"] = group_metrics

    return result

tabs = st.tabs([
    "Training",
    "Governance Report",
    "Bias & Fairness",
    "Stress Testing",
    "Monitoring",
    "Explainability",
    "Registry",
    "Audit Logs"
])
# 
=========================================================
# COMPARISON
# =========================================================

with tabs[1]:  # Governance Report

    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        model_name = st.selectbox(
            "Select Model for Governance Review",
            list(st.session_state.trained_models.keys()),
            key="gov_model_select"
        )

        metrics = st.session_state.leaderboard.get(model_name, {})

        st.subheader("Model Performance Summary")
        st.json(metrics)

        if st.button("Generate Governance Summary"):

            report = {
                "jurisdiction": jurisdiction,
                "model": model_name,
                "metrics": metrics,
                "generated_at": datetime.datetime.utcnow().isoformat()
            }

            report_str = json.dumps(report, indent=2)

            st.download_button(
                "Download Governance Report (JSON)",
                report_str,
                file_name="governance_report.json"
            )

            st.success("Governance report ready for audit submission.")

# =========================================================
# PREDICTION
# =========================================================
with tabs[2]:
    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        feature_names = st.session_state.feature_names
        st.subheader("🔮 Manual Prediction")
        inputs = []
        for f in feature_names:
            inputs.append(st.number_input(f, value=0.0, key=f"pred_input_{f}"))
        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys()),
            key="pred_model_selector"
        )
        if st.button("Predict", key="predict_btn"):
            model = st.session_state.trained_models[model_name]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            input_df = align_features(input_df, feature_names)
            pred = model.predict(input_df)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]
            st.success(f"Prediction: {pred}")
            # Log prediction
            log_prediction(
                input_df.to_dict(orient="records")[0],
                pred,
                prob,
                model_name
            )
            # ===== Production Drift Check =====
            try:
                ks_stat, ks_p = ks_2samp(
                    X_train[feature_names[0]],
                    X_test[feature_names[0]]
                )
                wasserstein_score = wasserstein_distance(
                    X_train[feature_names[0]],
                    X_test[feature_names[0]]
                )
                st.metric("Prediction Drift KS", f"{ks_p:.5f}")
                st.metric("Prediction Drift Wasserstein", f"{wasserstein_score:.5f}")
            except Exception:
                pass

# =========================================================
# MONITORING
# =========================================================
with tabs[3]:
    st.subheader("📊 Production Monitoring")
    st.write("Dataset Statistics")
    st.dataframe(X.describe())
    feature = st.selectbox(
        "Select Feature",
        feature_names,
        key="monitor_feature"
    )
    ks_stat, ks_p = ks_2samp(
        X_train[feature],
        X_test[feature]
    )
    wasserstein_score = wasserstein_distance(
        X_train[feature],
        X_test[feature]
    )
    st.metric("KS P Value", f"{ks_p:.5f}")
    st.metric("Wasserstein Distance", f"{wasserstein_score:.5f}")
    if ks_p < 0.05 or wasserstein_score > 0.1:
        st.warning("⚠️ Distribution Drift Risk Detected")
    else:
        st.success("✅ Stable Distribution")
    log_drift_metrics(
        feature,
        X_train[feature],
        X_test[feature],
        wasserstein_score,
        "production_monitoring"
    )

=========================================================
# EXPLAINABILITY
# =========================================================
with tabs[4]:  # Explainability
    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        st.subheader("Model Explainability")

        model_name = st.selectbox(
            "Select Model for Explainability",
            list(st.session_state.trained_models.keys()),
            key="explain_model_selectbox"
        )

        model = st.session_state.trained_models[model_name]

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            safe_barh(
                feature_names,
                model.feature_importances_,
                "Feature Importance"
            )

        # Linear models
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            safe_barh(
                feature_names,
                coefs,
                "Coefficient Importance"
            )

        # SHAP explanation
        if SHAP_AVAILABLE:
            if st.button("Run SHAP", key="shap_button"):
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

# =========================================================
# Stress Testing
# =========================================================
with tabs[3]:  # Stress Testing

    if not st.session_state.training_done:
        st.info("Train models first")
    else:
        model_name = st.selectbox(
            "Select Model for Stress Test",
            list(st.session_state.trained_models.keys()),
            key="stress_model_select"
        )

        model = st.session_state.trained_models[model_name]

        shock = st.slider("Macro Shock Factor (%)", -50, 50, 10)
        shock = shock / 100

        def stress_test(model, X, shock_factor):
            stressed_X = X.copy()
            stressed_X = stressed_X * (1 + shock_factor)
            preds = model.predict(stressed_X)
            return float(np.mean(preds))

        impact = stress_test(model, X_test, shock)

        st.metric("Predicted Default Rate Under Stress", f"{impact:.4f}")

        if impact > 0.5:
            st.warning("⚠️ Elevated systemic risk under stress scenario.")
        else:
            st.success("✅ Model stable under stress conditions.")
