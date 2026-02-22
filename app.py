# =========================================================
# JTYYLSPH V5 ENTERPRISE AI PLATFORM
# Production Hardened Edition
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime, json, pickle, os, io
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, roc_curve, auc
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
REGISTRY_SCHEMA_VERSION = "2.0"
ARTIFACT_DIR = "artifacts"
LOG_FILE = "prediction_logs.jsonl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# LOGIN
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
# SESSION STATE
# =========================================================

for key in ["trained_models", "best_model", "leaderboard", "training_done"]:
    if key not in st.session_state:
        st.session_state[
            key] = {} if key == "trained_models" else [] if key == "leaderboard" else None if key == "best_model" else False


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
    # Load or create registry
    registry = []
    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)

    version_tag = f"v{len(registry) + 1}"
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
# PAGE TITLE
# =========================================================

st.title("🚀 JTYYLSPH V5 ENTERPRISE AI PLATFORM")
st.caption("AutoML • MLOps • Drift Detection • Explainability • Audit Logging")

# =========================================================
# DATA UPLOAD / SYNTHETIC DATA
# =========================================================

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
domain = st.sidebar.selectbox("Synthetic Dataset", ["Finance", "Healthcare", "Sports", "General"])

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    n_samples = 500
    if domain == "Finance":
        X_data, y_data = make_classification(n_samples=n_samples, n_features=6, n_informative=4, random_state=42)
    elif domain == "Healthcare":
        X_data, y_data = make_classification(n_samples=n_samples, n_features=8, n_informative=5, random_state=1)
    elif domain == "Sports":
        X_data, y_data = make_classification(n_samples=n_samples, n_features=5, n_informative=3, random_state=7)
    else:
        X_data, y_data = make_classification(n_samples=400, n_features=4, random_state=0)
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

X.columns = [str(c) for c in X.columns]
feature_names = list(X.columns)
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

tabs = st.tabs(
    ["🤖 Training", "📊 Comparison", "🔮 Prediction", "📈 Monitoring", "⚖️ Fairness", "🗂 Registry", "📜 Audit Logs"])

# ---------------------------
# TRAINING TAB
# ---------------------------
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
        st.success("Training Complete")

    if st.session_state.training_done:
        st.subheader("Model Performance")
        df_lb = pd.DataFrame(st.session_state.leaderboard)
        st.dataframe(df_lb)
        fig, ax = plt.subplots()
        ax.bar(df_lb["Model"], df_lb["accuracy"])
        ax.set_title("Accuracy Comparison")
        st.pyplot(fig)
        # Confusion matrix for best model
        if st.session_state.best_model:
            preds = st.session_state.best_model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
            st.pyplot(fig_cm)

# ---------------------------
# PREDICTION TAB
# ---------------------------
with tabs[2]:
    st.header("Manual Prediction")
    if not st.session_state.trained_models:
        st.info("⚠️ Train models first")
    else:
        inputs = []
        for f in feature_names:
            val = st.number_input(label=f, value=float(X[f].mean()))
            inputs.append(val)
        model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
        if st.button("Predict"):
            model = st.session_state.trained_models[model_name]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {pred}")
            if prob is not None:
                st.metric("Confidence", f"{prob:.2f}")
                st.metric("Risk Score", f"{prob * 100:.1f}")
            log_prediction(input_df.to_dict(orient="records")[0], pred, prob, model_name)

# ---------------------------
# REGISTRY TAB
# ---------------------------
with tabs[5]:
    st.header("Model Registry")
    if os.path.exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY, "r") as f:
            registry = json.load(f)
        st.dataframe(pd.DataFrame(registry))
    else:
        st.info("No registry entries yet.")

# ---------------------------
# AUDIT LOG TAB
# ---------------------------
with tabs[6]:
    st.header("Prediction Audit Logs")
    if os.path.exists(LOG_FILE):
        logs = [json.loads(line) for line in open(LOG_FILE)]
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs)
        fig, ax = plt.subplots()
        df_logs["prediction"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No logs yet.")
# ---------------------------
# EXPLAINABILITY TAB
# ---------------------------
with tabs[4]:
    st.header("Explainability & Feature Importance")

    if not st.session_state.trained_models:
        st.info("⚠️ Train models first to use explainability features.")
    else:
        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys()),
            key="explain_model"
        )
        model = st.session_state.trained_models[model_name]

        st.subheader("SHAP Explainability")
        if SHAP_AVAILABLE:
            if st.button("Run SHAP", key="shap_button"):
                # Limit samples for performance
                sample_X = X_test.sample(min(100, len(X_test)), random_state=42)
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(sample_X)
                fig = plt.figure()
                shap.summary_plot(shap_values, sample_X, show=False)
                st.pyplot(fig)
        else:
            st.warning("SHAP not installed. Feature importance still available below.")

        st.subheader("Feature Importance / Coefficients")

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                fig, ax = plt.subplots()
                ax.barh(feature_names, importances)
                ax.set_xlabel("Importance")
                ax.set_title(f"Feature Importance — {model_name}")
                st.pyplot(fig)
            else:
                st.warning(f"Cannot plot feature importances: {len(importances)} != {len(feature_names)}")

        # Linear models (Logistic Regression)
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:  # Multi-class
                coefs = np.mean(np.abs(coefs), axis=0)
            coefs = np.atleast_1d(coefs)
            if len(coefs) == len(feature_names):
                fig, ax = plt.subplots()
                ax.barh(feature_names, coefs)
                ax.set_xlabel("Coefficient Magnitude")
                ax.set_title(f"Feature Coefficients — {model_name}")
                st.pyplot(fig)
            else:
                st.warning(f"Cannot plot coefficients: {len(coefs)} != {len(feature_names)}")
        else:
            st.info("Feature importance not available for this model type.")

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

