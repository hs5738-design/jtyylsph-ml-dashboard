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
import traceback

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ============================
# TORCH SETUP
# ============================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

# ============================
# SAFE CONSTANTS
# ============================
LOG_FILE = "logs.jsonl"
PREDICTION_DRIFT_LOG = "drift.jsonl"

# ============================
# GOVERNANCE MODEL
# ============================

if TORCH_AVAILABLE:
    class JTYYLSPHModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))


    def governance_loss(model, X, y, sensitive_idx=None):
        preds = model(X).squeeze()

        task_loss = nn.BCELoss()(preds, y)

        fairness_penalty = torch.tensor(0.0)
        if sensitive_idx is not None:
            s = X[:, sensitive_idx]
            thr = torch.median(s)
            g0 = preds[s <= thr]
            g1 = preds[s > thr]
            if len(g0) > 0 and len(g1) > 0:
                fairness_penalty = torch.abs(g0.mean() - g1.mean())

        drift_penalty = torch.abs(preds.mean() - y.mean())

        lambda_fair = 0.1
        lambda_drift = torch.clamp(drift_penalty * 2, 0, 1)

        loss = task_loss + lambda_fair * fairness_penalty + lambda_drift * drift_penalty

        return loss, task_loss.item(), fairness_penalty.item(), drift_penalty.item()


    def train_jtyylsph(X_train, y_train, sensitive_feature=None, epochs=30):
        X_tensor = torch.tensor(X_train.values.astype(np.float32))
        y_tensor = torch.tensor(y_train.values.astype(np.float32))

        model = JTYYLSPHModel(X_tensor.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        sensitive_idx = None
        if sensitive_feature in X_train.columns:
            sensitive_idx = list(X_train.columns).index(sensitive_feature)

        history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            loss, task_l, fair_l, drift_l = governance_loss(
                model, X_tensor, y_tensor, sensitive_idx
            )

            loss.backward()
            optimizer.step()

            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "task": task_l,
                "fairness": fair_l,
                "drift": drift_l,
            })

        return model, history


    def predict_jtyylsph(model, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X.values.astype(np.float32))
            preds = model(X_tensor).squeeze().numpy()
            return (preds > 0.5).astype(int)

# ============================
# STREAMLIT UI
# ============================

st.title("🚀 JTYYLSPH — Production AI Governance Platform")

# DATA
X_data, y_data = make_classification(n_samples=500, n_features=6, random_state=42)
X = pd.DataFrame(X_data)
y = pd.Series(y_data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODELS
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

# TRAIN BUTTON
if st.button("Train Models"):
    leaderboard = {}

    # GOVERNANCE MODEL
    if TORCH_AVAILABLE:
        model, history = train_jtyylsph(X_train, y_train, sensitive_feature=X.columns[0])
        preds = predict_jtyylsph(model, X_test)

        leaderboard["JTYYLSPH"] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        st.line_chart(pd.DataFrame(history).set_index("epoch"))

    # BASE MODELS
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        leaderboard[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

    st.success("Training Complete")
    st.dataframe(pd.DataFrame(leaderboard).T.sort_values("accuracy", ascending=False))

# ============================
# UTILITIES
# ============================

def log_json(file, entry):
    try:
        with open(file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass


def load_json(file):
    rows = []
    if os.path.exists(file):
        with open(file) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    continue
    return rows

# ============================
# MONITORING
# ============================

st.subheader("📊 Drift Monitoring")
feature = st.selectbox("Feature", X.columns)

train_vals = X_train[feature]
test_vals = X_test[feature]

drift = float(np.abs(train_vals.mean() - test_vals.mean()))

st.metric("Mean Drift", f"{drift:.5f}")

log_json(PREDICTION_DRIFT_LOG, {
    "time": datetime.datetime.utcnow().isoformat(),
    "feature": feature,
    "drift": drift
})

logs = load_json(PREDICTION_DRIFT_LOG)
if logs:
    df = pd.DataFrame(logs)
    if "drift" in df.columns:
        st.line_chart(df["drift"])

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

drift_penalty = torch.abs(pred_mean - model.running_mean)

model.running_mean = 0.9 * model.running_mean + 0.1 * pred_mean.detach()
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

torch.save(model.state_dict(), path)
