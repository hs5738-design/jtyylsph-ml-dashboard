# =========================================================
# JTYYLSPH V5 ENTERPRISE AI PLATFORM — V5.8 ELITE MERGE
# Production-Ready • Persistent Models • JSON Registry • Audit Logging
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle
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
# CONFIGURATION (V8 SAFE SYSTEM)
# =========================================================

MODEL_REGISTRY = "model_registry.json"
MODEL_DIR = "models"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# SAFE REGISTRY FUNCTIONS (V8)
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


def register_model(name, model, metrics):

    registry = load_registry()

    version = len(registry) + 1
    model_path = os.path.join(MODEL_DIR, f"{name}_v{version}.pkl")

    joblib.dump(model, model_path)

    record = {
        "name": name,
        "version": version,
        "path": model_path,
        "metrics": metrics,
        "time": datetime.datetime.utcnow().isoformat()
    }

    registry.append(record)
    save_registry(registry)


def load_models_from_registry():

    registry = load_registry()
    models = {}

    for rec in registry:
        try:
            path = rec.get("path")

            if not path or not os.path.exists(path):
                continue

            model = joblib.load(path)

            models[rec["name"]] = model

        except:
            pass

    return models


# =========================================================
# SESSION STATE
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

if "training_done" not in st.session_state:
    st.session_state.training_done = False

if "feature_names" not in st.session_state:
    st.session_state.feature_names = []


# =========================================================
# TITLE
# =========================================================

st.title("🚀 JTYYLSPH V5 ENTERPRISE AI PLATFORM")
st.caption("AutoML • MLOps • Drift Detection • Explainability • Audit Logging")


# =========================================================
# DATASET
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "General"]
)

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

else:
    if domain == "Finance":
        X_data, y_data = make_classification(
            n_samples=500, n_features=6, n_informative=4, random_state=42
        )
    elif domain == "Healthcare":
        X_data, y_data = make_classification(
            n_samples=500, n_features=8, n_informative=5, random_state=1
        )
    elif domain == "Sports":
        X_data, y_data = make_classification(
            n_samples=500, n_features=5, n_informative=3, random_state=7
        )
    else:
        X_data, y_data = make_classification(
            n_samples=400, n_features=4, random_state=0
        )

    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

X.columns = [str(c) for c in X.columns]

feature_names = list(X.columns)
st.session_state.feature_names = feature_names

st.write("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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
# LOGGING
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


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "🤖 Training",
    "📊 Model Comparison",
    "🔮 Prediction",
    "📈 Monitoring",
    "🧠 Explainability",
    "🗂 Registry",
    "📜 Audit Logs"
])


# =========================================================
# TRAINING
# =========================================================

with tabs[0]:

    st.header("AutoML Training")

    if st.button("Train All Models"):

        st.session_state.leaderboard = {}
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

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            }

            st.session_state.trained_models[name] = model
            st.session_state.leaderboard[name] = metrics

            register_model(name, model, metrics)

            if acc > best_score:
                best_score = acc

        st.session_state.training_done = True
        st.success("✅ Training Complete")


# =========================================================
# COMPARISON
# =========================================================

with tabs[1]:

    st.header("Model Performance Comparison")

    if st.session_state.training_done:

        df_lb = pd.DataFrame(st.session_state.leaderboard).T
        st.dataframe(df_lb)

        st.bar_chart(df_lb["accuracy"])

    else:
        st.info("Train models first.")


# =========================================================
# PREDICTION (V8 FIXED FEATURE ALIGNMENT)
# =========================================================

with tabs[2]:

    st.header("Manual Prediction")

    if not st.session_state.training_done:
        st.info("Train models first.")
    else:

        feature_names = st.session_state.feature_names

        inputs = []

        for f in feature_names:
            val = st.number_input(f, value=0.0)
            inputs.append(val)

        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys())
        )

        if st.button("Predict"):

            model = st.session_state.trained_models[model_name]

            input_df = pd.DataFrame([inputs], columns=feature_names)

            pred = model.predict(input_df)[0]

            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0][1]

            st.success(f"Prediction: {pred}")

            log_prediction(
                input_df.to_dict(orient="records")[0],
                pred,
                prob,
                model_name
            )


# =========================================================
# MONITORING
# =========================================================

with tabs[3]:

    st.header("Monitoring & Drift Detection")

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

    st.header("Model Explainability")

    if not st.session_state.training_done:
        st.info("Train models first.")

    else:

        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys()),
            key="explain"
        )

        model = st.session_state.trained_models[model_name]

        if hasattr(model, "feature_importances_"):

            fig, ax = plt.subplots()
            ax.barh(feature_names, model.feature_importances_)
            st.pyplot(fig)

        elif hasattr(model, "coef_"):

            coefs = np.abs(model.coef_[0])

            fig, ax = plt.subplots()
            ax.barh(feature_names, coefs)
            st.pyplot(fig)

        if SHAP_AVAILABLE:
            if st.button("Run SHAP"):
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test[:100])
                fig = plt.figure()
                shap.summary_plot(shap_values, X_test[:100], show=False)
                st.pyplot(fig)


# =========================================================
# REGISTRY
# =========================================================

with tabs[5]:

    st.header("Model Registry")

    reg = load_registry()

    if reg:
        st.dataframe(pd.DataFrame(reg))
    else:
        st.info("No registry entries yet.")


# =========================================================
# AUDIT LOGS
# =========================================================

with tabs[6]:

    st.header("Prediction Audit Logs")

    if os.path.exists(LOG_FILE):

        logs = [json.loads(line) for line in open(LOG_FILE)]
        st.dataframe(pd.DataFrame(logs))

    else:
        st.info("No logs yet.")
