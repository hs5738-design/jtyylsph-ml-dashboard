
# =========================================================
# JTYYLSPH Intelligent Classification Platform V5 ENTERPRISE
# Unified: V3 AutoML + V4.5 PRO MLOps Architecture
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
import io

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from scipy.stats import ks_2samp

# Optional Libraries
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# =========================================================
# ENTERPRISE LOGIN
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.sidebar.subheader("🔐 Enterprise Login")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if user == "admin" and pwd == "admin123":
            st.session_state.authenticated = True
        else:
            st.sidebar.error("Invalid credentials")

if not st.session_state.authenticated:
    login()
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

LOG_FILE = "prediction_logs.jsonl"
MODEL_REGISTRY = "model_registry.pkl"


# =========================================================
# UTILITIES
# =========================================================

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
    if os.path.exists(MODEL_REGISTRY):
        registry = pickle.load(open(MODEL_REGISTRY, "rb"))
    else:
        registry = []
    record = {
        "version": f"v{len(registry)+1}",
        "name": name,
        "timestamp": datetime.datetime.now(),
        "metrics": metrics,
        "model": model
    }
    registry.append(record)
    pickle.dump(registry, open(MODEL_REGISTRY, "wb"))

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
st.caption("AutoML • MLOps • Drift Detection • Fairness • Explainability • Deployment")


# =========================================================
# DATA
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
        X_data, y_data = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=42)
    elif domain == "Healthcare":
        X_data, y_data = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=1)
    elif domain == "Sports":
        X_data, y_data = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=7)
    else:
        X_data, y_data = make_classification(n_samples=400, n_features=4, random_state=0)
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

X.columns = [str(col) for col in X.columns]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================================================
# MODELS + PARAM GRIDS
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

param_grids = {
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 5]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
}


# =========================================================
# TABS
# =========================================================

tabs = st.tabs([
    "🤖 Training",
    "📊 Model Comparison",
    "🔮 Prediction",
    "📁 Batch Prediction",
    "📈 Monitoring",
    "⚖️ Fairness",
    "🧠 Explainability",
    "🗂 Registry",
    "🚀 API Export"
])


# =========================================================
# TRAINING TAB
# =========================================================

with tabs[0]:

    if st.button("Train All Models"):

        st.session_state.leaderboard = []
        best_score = 0

        for name, model in models.items():

            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            st.session_state.trained_models[name] = model

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")
            cv_scores = cross_val_score(model, X, y, cv=5)

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "cv_mean": cv_scores.mean()
            }

            st.session_state.leaderboard.append({"Model": name, **metrics})
            save_model_version(model, name, metrics)

            if acc > best_score:
                best_score = acc
                st.session_state.best_model = model

        # Stacking Ensemble
        estimators = [(n, m) for n, m in st.session_state.trained_models.items()]
        stack = StackingClassifier(estimators=estimators,
                                   final_estimator=LogisticRegression())
        stack.fit(X_train, y_train)
        st.session_state.trained_models["StackingEnsemble"] = stack

        st.success("Training Complete")


# =========================================================
# MODEL COMPARISON
# =========================================================

with tabs[1]:

    if st.session_state.leaderboard:

        df_lb = pd.DataFrame(st.session_state.leaderboard)
        st.dataframe(df_lb)

        fig, ax = plt.subplots()
        ax.bar(df_lb["Model"], df_lb["accuracy"])
        ax.set_title("Accuracy Comparison")
        st.pyplot(fig)

        # Confusion Matrix for best model
        model = st.session_state.best_model
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)


# =========================================================
# MANUAL PREDICTION
# =========================================================

with tabs[2]:

    if st.session_state.trained_models:

        inputs = [st.number_input(f, value=float(X[f].mean())) for f in feature_names]
        model_name = st.selectbox("Model", list(st.session_state.trained_models.keys()))

        if st.button("Predict"):
            model = st.session_state.trained_models[model_name]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            st.success(f"Prediction: {pred}")

            if prob is not None:
                st.metric("Confidence", f"{prob:.2f}")
                st.metric("Risk Score", f"{prob*100:.1f}")

            log_prediction(input_df.to_dict(orient="records")[0], pred, prob, model_name)


# =========================================================
# MONITORING
# =========================================================

with tabs[4]:

    feature = st.selectbox("Feature", feature_names)

    psi = population_stability_index(X_train[feature], X_test[feature])
    st.metric("PSI", round(psi,4))

    stat, p = ks_2samp(X_train[feature], X_test[feature])
    st.metric("KS p-value", round(p,4))


# =========================================================
# FAIRNESS
# =========================================================

with tabs[5]:

    if st.session_state.best_model:
        preds = st.session_state.best_model.predict(X_test)
        group = pd.qcut(X_test.iloc[:,0], 2, labels=["Low","High"])
        df_fair = pd.DataFrame({"group":group,"pred":preds})
        rates = df_fair.groupby("group")["pred"].mean()
        st.write("Group Positive Rates")
        st.write(rates)

        positive_rate_diff = abs(rates["High"] - rates["Low"])
        st.metric("Demographic Parity Difference", round(positive_rate_diff,4))


# =========================================================
# EXPLAINABILITY
# =========================================================

with tabs[6]:

    if SHAP_AVAILABLE and st.session_state.best_model:
        sample_X = X_test.sample(min(100, len(X_test)), random_state=42)
        explainer = shap.Explainer(st.session_state.best_model, X_train)
        shap_values = explainer(sample_X)
        fig = plt.figure()
        shap.summary_plot(shap_values, sample_X, show=False)
        st.pyplot(fig)


# =========================================================
# REGISTRY
# =========================================================

with tabs[7]:

    if os.path.exists(MODEL_REGISTRY):
        registry = pickle.load(open(MODEL_REGISTRY, "rb"))
        rows = [{"version":r["version"],"name":r["name"],"time":r["timestamp"],**r["metrics"]} for r in registry]
        st.dataframe(pd.DataFrame(rows))


# =========================================================
# API EXPORT
# =========================================================

with tabs[8]:

    if st.session_state.best_model:
        api_code = """
from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("best_model.pkl","rb"))

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
"""
        st.code(api_code, language="python")


