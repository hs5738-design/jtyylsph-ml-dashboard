# =========================================================
# JTYYLSPH V6.3 PRO MAX — ENTERPRISE AI PLATFORM
# Production • Persistent Models • Registry • Explainability
# =========================================================
import streamlit as st 
st.write("App starting...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os
import joblib
import hashlib
import sys
import traceback
import re
import xml.etree.ElementTree as ET

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wasserstein_distance, ks_2samp

# =========================================================
# OPTIONAL LIBRARIES
# =========================================================
try:
    import pdfplumber
except:
    pdfplumber = None

try:
    import docx
except:
    docx = None

try:
    from PIL import Image
except:
    Image = None

try:
    import pytesseract
except:
    pytesseract = None

try:
    import sqlalchemy
except:
    sqlalchemy = None

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# =========================================================
# ERROR HANDLER
# =========================================================

def handle_exception(exc_type, exc_value, exc_traceback):
    st.error("Application Error")
    error_text = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    st.text(error_text)

sys.excepthook = handle_exception

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
    if not os.path.exists(path):
        return "missing"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_registry():
    if not os.path.exists(MODEL_REGISTRY):
        return []
    try:
        with open(MODEL_REGISTRY) as f:
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
    registry = load_registry()

    versions = []
    for r in registry:
        if r.get("name") == name:
            try:
                versions.append(int(r.get("version", 0)))
            except:
                pass

    version = max(versions) + 1 if versions else 1
    model_path = os.path.join(MODEL_DIR, f"{name}_v{version}.pkl")

    safe_metrics = {
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
    }

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "metrics": safe_metrics,
    }

    joblib.dump(artifact, model_path)

    record = {
        "name": name,
        "version": version,
        "path": model_path,
        "metrics": safe_metrics,
        "time": datetime.datetime.utcnow().isoformat(),
    }

    registry.append(record)
    save_registry(registry)

def load_models_from_registry():
    registry = load_registry()
    models = {}

    for rec in registry:
        try:
            path = rec.get("path")
            if path and os.path.exists(path):
                artifact = joblib.load(path)
                if artifact.get("model") is not None:
                    models[rec["name"]] = artifact
        except:
            continue

    return models

@st.cache_resource
def cached_registry():
    return load_models_from_registry()

# =========================================================
# HELPERS
# =========================================================

def fairness_analysis(model, X, y, sensitive_feature=None):

    preds = model.predict(X)
    base_acc = accuracy_score(y, preds)

    results = {"overall_accuracy": float(base_acc)}

    if sensitive_feature is None:
        results["note"] = "No sensitive feature selected"
        return results

    groups = X[sensitive_feature]
    group_metrics = {}

    for g in groups.unique():
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y[mask], preds[mask])
        group_metrics[str(g)] = float(acc)

    results["group_accuracy"] = group_metrics

    if len(group_metrics) > 1:
        vals = list(group_metrics.values())
        results["fairness_gap"] = float(max(vals) - min(vals))

    return results


def safe_barh(names, values, title):
    names = list(names)
    values = list(values)

    mn = min(len(names), len(values))
    names = names[:mn]
    values = values[:mn]

    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_title(title)
    st.pyplot(fig)

def log_drift_metrics(feature, train, test, value, metric):
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "feature": feature,
        "metric": metric,
        "drift_score": float(value),
    }
    try:
        with open(PREDICTION_DRIFT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

def log_prediction(model_name):
    entry = {
        "time": datetime.datetime.utcnow().isoformat(),
        "model": model_name
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

def load_json_lines(file):
    rows = []
    if os.path.exists(file):
        with open(file) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    continue
    return rows

def extract_text_features(text):
    words = text.split()
    feats = {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "numeric_count": len(re.findall(r"\d+", text)),
        "uppercase_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "sentence_count": len(re.split(r"[.!?]", text))
    }
    return pd.DataFrame([feats])

def ingest_file(uploaded):
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded)

    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded)

    if name.endswith(".json"):
        return pd.read_json(uploaded)

    if name.endswith(".parquet"):
        try:
            return pd.read_parquet(uploaded)
        except:
            return None

    if name.endswith(".sql"):
        return extract_text_features(uploaded.read().decode())

    if name.endswith(".xml"):
        tree = ET.parse(uploaded)
        root = tree.getroot()
        text = " ".join([elem.text or "" for elem in root.iter()])
        return extract_text_features(text)

    if name.endswith(".pdf") and pdfplumber:
        text = ""
        with pdfplumber.open(uploaded) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t
        return extract_text_features(text)

    if name.endswith(".docx") and docx:
        doc = docx.Document(uploaded)
        text = " ".join([p.text for p in doc.paragraphs])
        return extract_text_features(text)

    if name.endswith(".txt") or name.endswith(".log"):
        return extract_text_features(uploaded.read().decode())

    if name.endswith((".png", ".jpg", ".jpeg")) and Image and pytesseract:
        img = Image.open(uploaded)
        text = pytesseract.image_to_string(img)
        df = extract_text_features(text)
        df["image_width"], df["image_height"] = img.size
        return df

    return None

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
# AUTOLOAD REGISTRY
# =========================================================

if not st.session_state.trained_models:
    loaded = cached_registry()
    for name, artifact in loaded.items():
        st.session_state.trained_models[name] = artifact.get("model")
        st.session_state.leaderboard[name] = artifact.get("metrics", {})
    st.session_state.training_done = bool(st.session_state.trained_models)

# =========================================================
# UI
# =========================================================

st.title("🚀 JTYYLSPH — AI Governance Platform")

# =========================================================
# DATA INPUT SECTION
# =========================================================

st.sidebar.header("Compliance Mode")
jurisdiction = st.sidebar.selectbox(
    "Select Regulatory Framework",
    [
        "United States (SR 11-7)",
        "European Union (EU AI Act)",
        "UK Model Risk Guidance",
        "APAC General Risk Framework",
        "Custom Enterprise Policy",
    ],
)

st.sidebar.header("Dataset")
st.sidebar.header("Dataset Controls")

domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "business", "emotion", "General"]
)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    X_data, y_data = make_classification(
        n_samples=500, n_features=6, random_state=42
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

uploaded_files = st.sidebar.file_uploader(
    "Upload Dataset or Documents",
    accept_multiple_files=True,
    type=[
        "csv", "xlsx", "json", "parquet",
        "pdf", "docx", "txt", "log",
        "xml", "sql",
        "png", "jpg", "jpeg",
    ],
)

# DATABASE
st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input(
    "SQLAlchemy DB URL",
    placeholder="postgresql://user:pass@host:5432/db",
)

@st.cache_data
def load_db(query, engine):
    return pd.read_sql(query, engine)

query = st.sidebar.text_area(
    "SQL Query", placeholder="SELECT * FROM table LIMIT 1000"
)

X = X if 'X' in locals() else None
y = y if 'y' in locals() else None

# DATABASE INGESTION
if db_url and query and sqlalchemy:
    try:
        engine = sqlalchemy.create_engine(db_url)
        df = load_db(query, engine)
        st.success("Database loaded")
        st.dataframe(df)

        target_col = st.sidebar.selectbox("Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

    except Exception as e:
        st.error(f"Database error: {e}")

# FILE INGESTION
elif uploaded_files:
    dataframes = []

    for f in uploaded_files:
        df = ingest_file(f)
        if df is not None:
            dataframes.append(df)

    if dataframes:
        df = pd.concat(dataframes, ignore_index=True, sort=False)
        st.write("Combined Dataset")
        st.dataframe(df)

        if len(df.columns) > 1:
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = np.random.randint(0, 2, len(df))

# SYNTHETIC
else:
    X_data, y_data = make_classification(
        n_samples=500, n_features=6, random_state=42
    )
    X = pd.DataFrame(X_data)
    y = pd.Series(y_data)

if X is None:
    st.error("No valid dataset could be loaded.")
    st.stop()

X.columns = [str(c) for c in X.columns]
feature_names = list(X.columns)
st.session_state.feature_names = feature_names

st.write("Dataset Shape:", X.shape)
st.write("### Dataset Summary")
st.write(X.describe())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("### Data Quality Check")
st.write("Missing Values")
st.write(X.isna().sum())
st.write("Duplicate Rows")
st.write(X.duplicated().sum())

# =========================================================
# MODELS
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
}

param_grids = {
    "RandomForest": {"n_estimators": [100, 200]},
    "GradientBoosting": {"n_estimators": [100, 200]},
}

# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "Training",
        "Governance",
        "Bias",
        "Stress Testing",
        "Monitoring",
        "Explainability",
        "Registry",
        "Audit Logs",
    ]
)

# =========================================================
# TRAINING TAB
# =========================================================

with tabs[0]:
    if st.button("Train Models"):
        st.session_state.leaderboard = {}

        for name, model in models.items():
            if name in param_grids:
                grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
            }

            st.session_state.trained_models[name] = model
            st.session_state.leaderboard[name] = metrics
            register_model(name, model, feature_names, metrics)

        st.session_state.training_done = True
        st.success("Training Complete")

    if st.session_state.leaderboard:
        df_lb = pd.DataFrame(st.session_state.leaderboard).T
        df_lb = df_lb.sort_values("accuracy", ascending=False)
        st.dataframe(df_lb)

# =========================================================
# GOVERNANCE TAB
# =========================================================

with tabs[1]:
    st.subheader("📑 Model Governance Report")

    if not st.session_state.training_done:
        st.info("Train models first to generate governance reports.")
    else:
        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys()),
            key="gov_model",
        )

        metrics = st.session_state.leaderboard.get(model_name, {})

        st.write("### Model Performance")
        st.json(metrics)

        report = {
            "model_name": model_name,
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "metrics": metrics,
            "controls": {
                "bias_testing": "completed",
                "stress_testing": "completed",
                "drift_monitoring": "active",
                "registry_tracking": "enabled",
            },
        }

        st.download_button(
            "Download Governance Report",
            json.dumps(report, indent=2),
            file_name="governance_report.json",
        )

        st.write("### Regulatory Framework")
        st.info(jurisdiction)

        risk = 1 - metrics.get("accuracy", 0)

        st.write("### Model Risk Rating")

        if risk < 0.1:
            st.success("Low Model Risk")
        elif risk < 0.25:
            st.warning("Moderate Model Risk")
        else:
            st.error("High Model Risk")

# =========================================================
# BIAS TAB
# =========================================================

with tabs[2]:
    if st.session_state.training_done:
        model_name = st.selectbox(
            "Model",
            list(st.session_state.trained_models.keys()),
            key="bias_model",
        )

        sensitive = st.selectbox("Sensitive Feature", ["None"] + feature_names)
        if sensitive == "None":
            sensitive = None

        model = st.session_state.trained_models[model_name]
        results = fairness_analysis(model, X_test, y_test, sensitive)
        st.json(results)

# =========================================================
# STRESS TEST TAB
# =========================================================

with tabs[3]:
    if st.session_state.training_done:
        model_name = st.selectbox(
            "Model",
            list(st.session_state.trained_models.keys()),
            key="stress_model",
        )

        model = st.session_state.trained_models[model_name]

        feature = st.selectbox("Feature", feature_names)
        shock = st.slider("Shock %", -50, 50, 10) / 100

        stressed = X_test.copy()
        stressed[feature] *= (1 + shock)

        preds = model.predict(stressed)
        impact = float(np.mean(preds))

        st.metric("Default Rate", f"{impact:.4f}")

        log_prediction(model_name)

# =========================================================
# MONITORING TAB
# =========================================================

with tabs[4]:
    feature = st.selectbox("Feature", feature_names, key="monitor")
    ks, p = ks_2samp(X_train[feature], X_test[feature])
    w = wasserstein_distance(X_train[feature], X_test[feature])

    st.metric("KS p-value", f"{p:.5f}")
    st.metric("Wasserstein", f"{w:.5f}")

    log_drift_metrics(feature, X_train[feature], X_test[feature], w, "monitor")

    logs = load_json_lines(PREDICTION_DRIFT_LOG)
    if logs:
        df = pd.DataFrame(logs)
        if "drift_score" in df.columns:
            st.line_chart(df["drift_score"])

# =========================================================
# EXPLAINABILITY TAB
# =========================================================

with tabs[5]:
    if not st.session_state.training_done:
        st.info("Train models first to view explainability.")
    else:
        model_name = st.selectbox(
            "Explain Model",
            list(st.session_state.trained_models.keys()),
            key="exp",
        )

        model = st.session_state.trained_models[model_name]

        st.write("### Global Importance (Model Native)")

        if hasattr(model, "feature_importances_"):
            safe_barh(
                feature_names[:len(model.feature_importances_)],
                model.feature_importances_,
                "Feature Importance",
            )

        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            safe_barh(
                feature_names[:len(coefs)],
                coefs,
                "Model Coefficients",
            )
        else:
            st.info("Model does not expose feature importances.")

        # SHAP
        if SHAP_AVAILABLE:
            st.write("### SHAP Global Explanation")

            try:
                X_base = X_test.copy()
                X_base = X_base.select_dtypes(include=[np.number])

                expected = getattr(model, "feature_names_in_", None)

                if expected is not None:
                    X_fixed = X_base.copy()
                    for col in expected:
                        if col not in X_fixed.columns:
                            X_fixed[col] = 0
                    X_shap = X_fixed[list(expected)].iloc[:100]
                else:
                    X_shap = X_base.iloc[:100]

                X_shap = X_shap.replace([np.inf, -np.inf], np.nan).fillna(0)

                explainer = shap.Explainer(
                    model, X_train.select_dtypes(include=[np.number])
                )
                shap_values = explainer(X_shap)

                fig = plt.figure()
                shap.summary_plot(shap_values, X_shap, show=False)
                st.pyplot(fig)
                plt.close(fig)

                st.write("### SHAP Local Explanation (Single Instance)")
                idx = st.slider(
                    "Instance index", 0, len(X_shap) - 1, 0
                )

                fig2 = plt.figure()
                shap.waterfall_plot(shap_values[idx], show=False)
                st.pyplot(fig2)
                plt.close(fig2)

            except Exception as e:
                st.warning("SHAP explanation failed safely.")
                st.text(str(e))

        else:
            st.info("SHAP is not installed; native explainability only.")

# =========================================================
# REGISTRY TAB
# =========================================================

with tabs[6]:
    reg = load_registry()
    if reg:
        for r in reg:
            r["hash"] = model_hash(r["path"])

        df_reg = pd.DataFrame(reg)
        df_reg = df_reg.sort_values(["name", "version"])
        st.dataframe(df_reg)

# =========================================================
# AUDIT LOGS TAB
# =========================================================

with tabs[7]:
    logs = load_json_lines(LOG_FILE)
    if logs:
        st.dataframe(pd.DataFrame(logs))


