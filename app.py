# =========================================================
# JTYYLSPH V6 PRO ENTERPRISE AI PLATFORM
# Full MLOps • Registry Lifecycle • Batch Prediction • SHAP
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
import datetime

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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

ARTIFACT_DIR = "artifacts"
REGISTRY_FILE = "model_registry.json"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# SESSION INIT
# =========================================================

if "models" not in st.session_state:
    st.session_state.models = {}

if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

if "best_model" not in st.session_state:
    st.session_state.best_model = None

# =========================================================
# UTILITIES
# =========================================================

def save_registry(record):
    if os.path.exists(REGISTRY_FILE):
        registry = json.load(open(REGISTRY_FILE))
    else:
        registry = []

    registry.append(record)
    json.dump(registry, open(REGISTRY_FILE, "w"), indent=2)


def save_model(model, name, metrics):

    version = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = f"{ARTIFACT_DIR}/{name}_{version}.pkl"

    with open(path, "wb") as f:
        pickle.dump(model, f)

    record = {
        "name": name,
        "version": version,
        "stage": "Dev",
        "metrics": metrics,
        "path": path,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    save_registry(record)


def load_models_from_registry():
    if not os.path.exists(REGISTRY_FILE):
        return

    registry = json.load(open(REGISTRY_FILE))

    for rec in registry:
        path = rec["path"]
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.session_state.models[rec["name"]] = pickle.load(f)


def log_prediction(data, pred, model_name):
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "input": data,
        "prediction": int(pred),
        "model": model_name
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


load_models_from_registry()

# =========================================================
# UI
# =========================================================

st.title("🚀 JTYYLSPH V6 PRO Enterprise AI Platform")

tabs = st.tabs([
    "🤖 Training",
    "📊 Comparison",
    "🔮 Prediction",
    "📂 Batch",
    "📈 Monitoring",
    "🧠 Explainability",
    "⚖️ Fairness",
    "🗂 Registry",
    "📜 Audit"
])

# =========================================================
# DATA
# =========================================================

X_data, y_data = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=4,
    random_state=42
)

X = pd.DataFrame(X_data)
y = pd.Series(y_data)

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# TRAINING
# =========================================================

models_config = {
    "RandomForest": (RandomForestClassifier(), {"n_estimators": [100, 200]}),
    "GradientBoosting": (GradientBoostingClassifier(), {"n_estimators": [100, 200]}),
    "LogisticRegression": (LogisticRegression(max_iter=1000), None)
}

with tabs[0]:

    st.header("AutoML Training")

    if st.button("Train Models"):

        leaderboard = []
        best_acc = 0

        for name, (model, grid) in models_config.items():

            if grid:
                gs = GridSearchCV(model, grid, cv=3)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            st.session_state.models[name] = model

            leaderboard.append({
                "Model": name,
                "Accuracy": acc
            })

            save_model(model, name, {"accuracy": acc})

            if acc > best_acc:
                best_acc = acc
                st.session_state.best_model = model

        st.session_state.leaderboard = leaderboard
        st.success("Training Complete")

# =========================================================
# COMPARISON
# =========================================================

with tabs[1]:

    if st.session_state.leaderboard:
        df = pd.DataFrame(st.session_state.leaderboard)
        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.bar(df["Model"], df["Accuracy"])
        st.pyplot(fig)

# =========================================================
# PREDICTION
# =========================================================

with tabs[2]:

    if not st.session_state.models:
        st.warning("Train models first")
    else:

        inputs = {}

        for f in feature_names:
            inputs[f] = st.number_input(f, value=0.0)

        model_name = st.selectbox(
            "Model",
            list(st.session_state.models.keys())
        )

        if st.button("Predict"):

            model = st.session_state.models[model_name]

            input_df = pd.DataFrame([inputs])[feature_names]

            pred = model.predict(input_df)[0]

            st.success(f"Prediction: {pred}")

            log_prediction(inputs, pred, model_name)

# =========================================================
# BATCH PREDICTION
# =========================================================

with tabs[3]:

    st.header("Batch Prediction")

    file = st.file_uploader("Upload CSV for batch prediction")

    if file and st.session_state.models:

        df_batch = pd.read_csv(file)

        model_name = st.selectbox(
            "Model for Batch",
            list(st.session_state.models.keys()),
            key="batch"
        )

        if st.button("Run Batch"):

            model = st.session_state.models[model_name]

            preds = model.predict(df_batch[feature_names])

            df_batch["prediction"] = preds

            st.dataframe(df_batch)

            csv = df_batch.to_csv(index=False).encode()

            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv"
            )

# =========================================================
# MONITORING
# =========================================================

with tabs[4]:

    st.header("Monitoring")

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
        st.warning("Drift detected")
    else:
        st.success("No drift")

# =========================================================
# EXPLAINABILITY
# =========================================================

with tabs[5]:

    if not st.session_state.models:
        st.warning("Train models first")

    else:

        model_name = st.selectbox(
            "Model",
            list(st.session_state.models.keys()),
            key="exp"
        )

        model = st.session_state.models[model_name]

        if SHAP_AVAILABLE:

            if st.button("Run SHAP"):

                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test[:100])

                fig = plt.figure()
                shap.summary_plot(shap_values, X_test[:100], show=False)
                st.pyplot(fig)

        if hasattr(model, "feature_importances_"):

            imp = model.feature_importances_

            fig, ax = plt.subplots()
            ax.barh(feature_names, imp)
            st.pyplot(fig)

# =========================================================
# FAIRNESS
# =========================================================

with tabs[6]:

    st.header("Fairness Analysis")

    group = st.selectbox("Group Feature", feature_names)

    if st.session_state.models:

        model_name = st.selectbox(
            "Model",
            list(st.session_state.models.keys()),
            key="fair"
        )

        model = st.session_state.models[model_name]

        preds = model.predict(X_test)

        df_fair = X_test.copy()
        df_fair["target"] = y_test.values
        df_fair["pred"] = preds

        group_acc = df_fair.groupby(group).apply(
            lambda d: accuracy_score(d["target"], d["pred"])
        )

        st.write(group_acc)

# =========================================================
# REGISTRY
# =========================================================

with tabs[7]:

    st.header("Model Registry")

    if os.path.exists(REGISTRY_FILE):

        registry = json.load(open(REGISTRY_FILE))
        df = pd.DataFrame(registry)

        st.dataframe(df)

# =========================================================
# AUDIT
# =========================================================

with tabs[8]:

    st.header("Prediction Logs")

    if os.path.exists(LOG_FILE):

        logs = [json.loads(l) for l in open(LOG_FILE)]
        st.dataframe(pd.DataFrame(logs))
