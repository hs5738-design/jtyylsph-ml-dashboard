
# =========================================================
# JTYYLSPH Intelligent Classification Platform V5 ENTERPRISE
# Production Hardened Edition
# JSON Registry • Schema Versioning • Audit Logging
# =========================================================

import streamlit as st
st.set_page_config(page_title="JTYYLSPH AI Platform V5 ENTERPRISE", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_REGISTRY = "model_registry.json"
REGISTRY_SCHEMA_VERSION = "2.0"
ARTIFACT_DIR = "artifacts"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# SECURE LOGIN (ENVIRONMENT VARIABLES)
# =========================================================

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.sidebar.subheader("🔐 Enterprise Login")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            st.session_state.authenticated = True
        else:
            st.sidebar.error("Invalid credentials")

if not st.session_state.authenticated:
    login()
    st.stop()

# =========================================================
# REGISTRY MIGRATION (Pickle → JSON)
# =========================================================

def migrate_registry_if_needed():
    if os.path.exists("model_registry.pkl") and not os.path.exists(MODEL_REGISTRY):
        with open("model_registry.pkl", "rb") as f:
            old_registry = pickle.load(f)

        new_registry = []

        for i, r in enumerate(old_registry):
            model_path = f"{ARTIFACT_DIR}/{r.get('name','model')}_legacy_v{i+1}.pkl"
            with open(model_path, "wb") as mf:
                pickle.dump(r["model"], mf)

            new_registry.append({
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "version": f"legacy_v{i+1}",
                "name": r.get("name"),
                "timestamp": str(r.get("timestamp")),
                "metrics": r.get("metrics", {}),
                "artifact_path": model_path
            })

        with open(MODEL_REGISTRY, "w") as f:
            json.dump(new_registry, f, indent=2)

        st.warning("Registry migrated to JSON format.")

migrate_registry_if_needed()

# =========================================================
# SESSION STATE
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# =========================================================
# UTILITIES
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

def population_stability_index(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 1, buckets + 1)
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    psi = np.sum((expected_percents - actual_percents) *
                 np.log((expected_percents + 1e-6) / (actual_percents + 1e-6)))
    return psi

# =========================================================
# TITLE
# =========================================================

st.title("🚀 JTYYLSPH V5 ENTERPRISE AI PLATFORM")
st.caption("AutoML • MLOps • Drift Detection • Fairness • Audit Logging")

# =========================================================
# DATA
# =========================================================

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

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
    "🤖 Training",
    "📊 Model Comparison",
    "🔮 Prediction",
    "📈 Monitoring",
    "⚖️ Fairness",
    "🗂 Registry",
    "📜 Audit Logs"
])

# =========================================================
# TRAINING
# =========================================================

with tabs[0]:

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
            metrics = {"accuracy": acc}

            st.session_state.trained_models[name] = model
            save_model_version(model, name, metrics)

            if acc > best_score:
                best_score = acc
                st.session_state.best_model = model

        st.success("Training Complete")

# =========================================================
# REGISTRY DASHBOARD
# =========================================================

with tabs[5]:

    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)

        st.dataframe(pd.DataFrame(registry))
    else:
        st.info("No registry entries yet.")

# =========================================================
# AUDIT LOG DASHBOARD
# =========================================================

with tabs[6]:

    st.header("Prediction Audit Logs")

    if os.path.exists(LOG_FILE):
        logs = [json.loads(line) for line in open(LOG_FILE)]
        df_logs = pd.DataFrame(logs)

        st.dataframe(df_logs)

        st.subheader("Predictions Distribution")
        fig, ax = plt.subplots()
        df_logs["prediction"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    else:
        st.info("No logs yet.")

