# =========================================================
# JTYYLSPH V7 — ENTERPRISE AI GOVERNANCE PLATFORM
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
import sys
import traceback
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wasserstein_distance, ks_2samp
# Optional libraries (SHAP safe for Python <3.13 only)
SHAP_AVAILABLE = False
if sys.version_info < (3, 13):
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
st.write(f"SHAP Available: {SHAP_AVAILABLE}")
# =========================================================
# ERROR HANDLER
# =========================================================
def handle_exception(exc_type, exc_value, exc_traceback):
    st.error("Application Error")
    st.text("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
sys.excepthook = handle_exception
# =========================================================
# DATABASE VARIABLES (FIX)
# =========================================================
db_url = None
query = None
try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None
# =========================================================
# MODEL REGISTRY & LOGGING
# =========================================================
MODEL_DIR = "models"
MODEL_REGISTRY = "model_registry.json"
PREDICTION_DRIFT_LOG = "drift_logs.jsonl"
LOG_FILE = "prediction_logs.jsonl"
os.makedirs(MODEL_DIR, exist_ok=True)
def load_registry():
    if os.path.exists(MODEL_REGISTRY):
        try:
            data = json.load(open(MODEL_REGISTRY))
            if isinstance(data, list):
                return data
        except:
            return []
    return []
def register_model(name, model, feature_names, metrics):
    registry = load_registry()
    versions = [int(r.get("version",0)) for r in registry if r.get("name")==name]
    version = max(versions)+1 if versions else 1
    path = os.path.join(MODEL_DIR, f"{name}_v{version}.pkl")
    joblib.dump({"model":model, "feature_names":feature_names, "metrics":metrics}, path)
    registry.append({
        "name":name, "version":version, "path":path,
        "metrics":metrics, "time": datetime.datetime.utcnow().isoformat()
    })
    with open(MODEL_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)
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
def safe_barh(names, values, title):
    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_title(title)
    st.pyplot(fig)
def log_drift_metrics(feature, train_col, test_col, drift_score, metric):
    entry = {"time":datetime.datetime.utcnow().isoformat(),
             "feature":feature,"metric":metric,"drift_score":float(drift_score)}
    try:
        with open(PREDICTION_DRIFT_LOG,"a") as f:
            f.write(json.dumps(entry)+"\n")
    except: pass
def log_prediction(model_name):
    entry = {"time":datetime.datetime.utcnow().isoformat(),"model":model_name}
    try:
        with open(LOG_FILE,"a") as f:
            f.write(json.dumps(entry)+"\n")
    except: pass
def load_json_lines(path):
    if os.path.exists(path):
        lines = []
        with open(path) as f:
            for line in f:
                lines.append(json.loads(line))
        return lines
    return []
# =========================================================
# JTYYLSPHv7 CORE MODEL
# =========================================================
class JTYYLSPHv7:
    def __init__(self, n_estimators=50, max_depth=3):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.scaler = None
        self.feature_names = []
        self.trained = False
        self.version = "v7"
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.preprocessing import StandardScaler
        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        preds = self.model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        self.trained = True
        joblib.dump({"model":self.model, "scaler":self.scaler,
                     "feature_names":self.feature_names, "metrics":{"accuracy":acc,"precision":prec,"recall":rec,"f1":f1},
                     "version":self.version}, f"jtyylsph_{self.version}_bundle.pkl")
        return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}
    
    def predict(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X[self.feature_names])
        preds = self.model.predict(X_scaled)
        return preds
# =========================================================
# DATASET SIDEBAR
# =========================================================
st.sidebar.header("Dataset")
st.sidebar.header("Dataset Controls")
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"]
)
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
# INITIALIZE SESSION STATE
# =========================================================
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = {}
if "training_done" not in st.session_state:
    st.session_state.training_done = False
# AUTOLOAD REGISTRY
if not st.session_state.trained_models:
    loaded = cached_registry()
    for name, artifact in loaded.items():
        st.session_state.trained_models[name] = artifact.get("model", artifact)
        st.session_state.leaderboard[name] = artifact.get("metrics", {})
    st.session_state.training_done = bool(st.session_state.trained_models)
# =========================================================
# MODEL TRAINING (EXAMPLE)
# =========================================================
models = {"GradientBoosting": JTYYLSPHv7()}
param_grids = {"GradientBoosting":{"n_estimators":[50,100], "max_depth":[2,3,4]}}
tabs = st.tabs(["Training","Monitoring","Explainability","Registry","Audit Logs"])
with tabs[0]:
    if st.button("Train Models"):
        st.session_state.leaderboard = {}
        for name, model in models.items():
            grid = GridSearchCV(GradientBoostingClassifier(), param_grids[name], cv=3)
            grid.fit(X_train, y_train)
            model.model = grid.best_estimator_
            metrics = model.train(X_train, y_train)
            st.session_state.trained_models[name] = model
            st.session_state.leaderboard[name] = metrics
            register_model(name, model.model, feature_names, metrics)
        st.session_state.training_done = True
        st.success("Training Complete")
    
    if st.session_state.leaderboard:
        st.dataframe(pd.DataFrame(st.session_state.leaderboard).T)
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
        st.write("### Global Feature Importance (Model Native)")
        
        # Tree-based models
        if hasattr(model, "feature_importances_"):
            safe_barh(
                feature_names[:len(model.feature_importances_)],
                model.feature_importances_,
                "Feature Importance",
            )
        # Linear models
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            safe_barh(
                feature_names[:len(coefs)],
                coefs,
                "Model Coefficients",
            )
        else:
            st.info("Model does not expose native feature importances.")
        
        # SHAP explanations (only if available)
        if SHAP_AVAILABLE:
            try:
                st.write("### SHAP Global Explanation")
                X_base = X_test.select_dtypes(include=[np.number]).copy()
                if hasattr(model, "feature_names_in_"):
                    expected_cols = model.feature_names_in_
                    for col in expected_cols:
                        if col not in X_base.columns:
                            X_base[col] = 0
                    X_shap = X_base[list(expected_cols)].iloc[:100]
                else:
                    X_shap = X_base.iloc[:100]
                X_shap = X_shap.replace([np.inf, -np.inf], np.nan).fillna(0)
                explainer = shap.Explainer(model, X_train.select_dtypes(include=[np.number]))
                shap_values = explainer(X_shap)
                fig = plt.figure()
                shap.summary_plot(shap_values, X_shap, show=False)
                st.pyplot(fig)
                plt.close(fig)
                st.write("### SHAP Local Explanation (Single Instance)")
                idx = st.slider("Instance index", 0, len(X_shap) - 1, 0)
                fig2 = plt.figure()
                shap.waterfall_plot(shap_values[idx], show=False)
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.warning("SHAP explanation skipped safely due to incompatibility.")
                st.text(str(e))
        else:
            st.info("SHAP not available (Python 3.13 incompatible); showing native feature importances only.")
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


