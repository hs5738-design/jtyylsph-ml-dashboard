# =========================================================
# JTYYLSPH V6.3 PRO MAX — ENTERPRISE AI PLATFORM
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
import hashlib
import sys
import traceback
import re
import xml.etree.ElementTree as ET
import logging
logging.basicConfig(level=logging.DEBUG)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wasserstein_distance, ks_2samp

# Optional libraries
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

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE_V63 = True
except:
    TORCH_AVAILABLE_V63 = False
    torch = None
    nn = None
    optim = None

# =========================================================
# ERROR HANDLER
# =========================================================
def handle_exception(exc_type, exc_value, exc_traceback):
    st.error("Application Error")
    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
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
# PYTORCH MODEL CLASS
# =========================================================
if TORCH_AVAILABLE_V63:
    class JTYYLSPHModel_V63(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))

# =========================================================
# MODEL REGISTRY FUNCTIONS
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
        with open(MODEL_REGISTRY, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def save_registry(reg):
    with open(MODEL_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)

def register_model(name, model, feature_names, metrics):
    registry = load_registry()
    versions = [int(r.get("version",0)) for r in registry if r.get("name") == name]
    version = max(versions) + 1 if versions else 1
    is_torch = hasattr(model, "state_dict")
    model_path = os.path.join(MODEL_DIR, f"{name}_v{version}{'.pt' if is_torch else '.pkl'}")
    if is_torch:
        torch.save(model.state_dict(), model_path)
    else:
        joblib.dump({"model": model, "feature_names": feature_names, "metrics": metrics}, model_path)
    registry.append({
        "name": name,
        "version": version,
        "path": model_path,
        "metrics": metrics,
        "feature_names": feature_names,
        "time": datetime.datetime.utcnow().isoformat(),
        "type": "torch" if is_torch else "sklearn"
    })
    save_registry(registry)

def load_models_from_registry():
    registry = load_registry()
    models = {}
    for rec in registry:
        try:
            path = rec.get("path")
            if not path or not os.path.exists(path):
                continue
            if rec.get("type") == "torch":
                if not TORCH_AVAILABLE_V63:
                    continue
                feature_names = rec.get("feature_names", [])
                input_dim = len(feature_names)
                if input_dim == 0:
                    continue
                model = JTYYLSPHModel_V63(input_dim)
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
                models[f"{rec['name']}_v{rec['version']}"] = {"model": model, "feature_names": feature_names, "metrics": rec.get("metrics",{})}
            else:
                artifact = joblib.load(path)
                models[f"{rec['name']}_v{rec['version']}"] = artifact
        except Exception as e:
            print(f"Failed to load model {rec.get('name')}: {e}")
    return models

@st.cache_resource
def cached_registry():
    return load_models_from_registry()

# =========================================================
# HELPERS
# =========================================================
def fairness_analysis(model, X, y, sensitive_feature=None):
    preds = model.predict(X)
    results = {"overall_accuracy": float(accuracy_score(y, preds))}
    if sensitive_feature is None:
        results["note"] = "No sensitive feature selected"
        return results
    groups = X[sensitive_feature]
    group_metrics = {}
    for g in groups.unique():
        mask = (groups == g)
        if mask.sum() == 0: continue
        acc = accuracy_score(y[mask], preds[mask])
        group_metrics[str(g)] = float(acc)
    results["group_accuracy"] = group_metrics
    if len(group_metrics) > 1:
        vals = list(group_metrics.values())
        results["fairness_gap"] = float(max(vals)-min(vals))
    return results

def safe_barh(names, values, title):
    names = list(names)[:len(values)]
    values = list(values)[:len(names)]
    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_title(title)
    st.pyplot(fig)

def log_drift_metrics(feature, train, test, value, metric):
    entry = {"time": datetime.datetime.utcnow().isoformat(), "feature": feature, "metric": metric, "drift_score": float(value)}
    try:
        with open(PREDICTION_DRIFT_LOG,"a") as f:
            f.write(json.dumps(entry)+"\n")
    except: pass

def log_prediction(model_name):
    entry = {"time": datetime.datetime.utcnow().isoformat(), "model": model_name}
    try:
        with open(LOG_FILE,"a") as f:
            f.write(json.dumps(entry)+"\n")
    except: pass

def load_json_lines(file):
    rows = []
    if os.path.exists(file):
        with open(file) as f:
            for line in f:
                try: rows.append(json.loads(line))
                except: continue
    return rows

def extract_text_features(text):
    words = text.split()
    feats = {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "numeric_count": len(re.findall(r"\d+", text)),
        "uppercase_ratio": sum(c.isupper() for c in text)/max(len(text),1),
        "digit_ratio": sum(c.isdigit() for c in text)/max(len(text),1),
        "sentence_count": len(re.split(r"[.!?]", text))
    }
    return pd.DataFrame([feats])

def ingest_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"): return pd.read_csv(uploaded)
    if name.endswith(".xlsx"): return pd.read_excel(uploaded)
    if name.endswith(".json"): return pd.read_json(uploaded)
    if name.endswith(".parquet"):
        try: return pd.read_parquet(uploaded)
        except: return None
    if name.endswith(".sql"): return extract_text_features(uploaded.read().decode())
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
                if t: text += t
        return extract_text_features(text)
    if name.endswith(".docx") and docx:
        doc = docx.Document(uploaded)
        text = " ".join([p.text for p in doc.paragraphs])
        return extract_text_features(text)
    if name.endswith(".txt") or name.endswith(".log"):
        return extract_text_features(uploaded.read().decode())
    if name.endswith((".png",".jpg",".jpeg")) and Image and pytesseract:
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
        st.session_state.trained_models[name] = artifact["model"]
        st.session_state.leaderboard[name] = artifact.get("metrics",{})
    st.session_state.training_done = bool(st.session_state.trained_models)

# =========================================================
# UI HEADER
# =========================================================
st.title("🚀 JTYYLSPH — AI Governance Platform")

# =========================================================
# DATA INPUT (CSV / Synthetic)
# =========================================================
st.sidebar.header("Dataset")
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
st.session_state.feature_names = feature_names
st.write("Dataset Shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# MODELS
# =========================================================
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear")
}
param_grids = {
    "RandomForest": {"n_estimators":[100,200]},
    "GradientBoosting": {"n_estimators":[100,200]}
}

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["Training","Governance","Bias","Stress Testing","Monitoring","Explainability","Registry","Audit Logs"])
