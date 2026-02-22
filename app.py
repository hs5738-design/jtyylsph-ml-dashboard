# =========================================================
# JTYYLSPH Intelligent Classification Platform V2
# Production Architecture
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import pickle

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from scipy.stats import ks_2samp
import shap

# =========================================================
# Page Config
# =========================================================

st.set_page_config(page_title="JTYYLSPH AI Platform", layout="wide")
st.title("🚀 Intelligent Classification & Analytics Platform")

# =========================================================
# Logging
# =========================================================

def log_prediction(data, pred, prob, model_name):

    entry = {
        "time": datetime.datetime.now().isoformat(),
        "input": data,
        "prediction": int(pred),
        "probability": float(prob) if prob else None,
        "model": model_name,
    }

    with open("prediction_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# =========================================================
# Sidebar Dataset
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

domain = st.sidebar.selectbox(
    "Synthetic Domain",
    ["Finance", "Healthcare", "Sports", "General"]
)

if uploaded:

    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

else:

    if domain == "Finance":
        X, y = make_classification(
            n_samples=500,
            n_features=6,
            n_informative=4,
            random_state=42,
        )

    elif domain == "Healthcare":
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_informative=5,
            random_state=1,
        )

    elif domain == "Sports":
        X, y = make_classification(
            n_samples=500,
            n_features=5,
            n_informative=3,
            random_state=7,
        )

    else:
        X, y = make_classification(
            n_samples=400,
            n_features=4,
            random_state=0,
        )

    X = pd.DataFrame(X)
    y = pd.Series(y)

st.write("Dataset Shape:", X.shape)

# =========================================================
# Train Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

feature_names = list(X.columns)

# =========================================================
# Models + Hyperparameters
# =========================================================

models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
    },
}

leaderboard = []
trained_models = {}
best_model = None
best_score = 0

# =========================================================
# Tabs
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["🤖 Training", "🔮 Prediction", "📊 Analytics", "🧠 Explainability"]
)

# =========================================================
# TRAINING
# =========================================================

with tab1:

    st.header("Model Training & AutoML")

    if st.button("Train Models"):

        for name, model in models.items():

            # GridSearch if params exist
            if name in param_grids:

                grid = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=3,
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)
                model = grid.best_estimator_

            else:
                model.fit(X_train, y_train)

            trained_models[name] = model

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1 = f1_score(y_test, preds, average="weighted")

            cv_scores = cross_val_score(model, X, y, cv=5)

            leaderboard.append({
                "Model": name,
                "Accuracy": acc,
                "CV Mean": cv_scores.mean(),
            })

            # Best model tracking
            if acc > best_score:
                best_score = acc
                best_model = model

            st.subheader(name)
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1: {f1:.3f}")
            st.write(f"CV Mean: {cv_scores.mean():.3f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
            st.pyplot(fig_cm)

            # ROC Curve
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.legend()
                st.pyplot(fig)

        st.success("Training Complete")

        st.subheader("Leaderboard")
        st.dataframe(pd.DataFrame(leaderboard))


# =========================================================
# PREDICTION
# =========================================================

with tab2:

    st.header("Manual Prediction")

    inputs = []

    for f in feature_names:
        val = st.number_input(f, value=0.0)
        inputs.append(val)

    model_name = st.selectbox(
        "Model",
        list(models.keys())
    )

    if st.button("Predict"):

        model = trained_models.get(model_name)

        if model is None:
            st.warning("Train models first.")
        else:

            input_df = pd.DataFrame([inputs], columns=feature_names)

            pred = model.predict(input_df)[0]

            prob = (
                model.predict_proba(input_df)[0][1]
                if hasattr(model, "predict_proba")
                else None
            )

            st.success(f"Prediction: {pred}")

            if prob:
                st.metric("Confidence", f"{prob:.2f}")
                st.metric("Risk Score", f"{prob*100:.1f}")

            log_prediction(
                input_df.to_dict(),
                pred,
                prob,
                model_name
            )


# =========================================================
# ANALYTICS
# =========================================================

with tab3:

    st.header("Exploratory Analysis")

    st.write(X.describe())

    corr = X.corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    plt.colorbar(cax)
    st.pyplot(fig)

    # Drift detection
    st.subheader("Drift Detection")

    col = st.selectbox("Feature", feature_names)

    stat, p = ks_2samp(X_train[col], X_test[col])

    if p < 0.05:
        st.warning("⚠️ Possible Drift Detected")
    else:
        st.success("No significant drift")


# =========================================================
# EXPLAINABILITY
# =========================================================

with tab4:

    st.header("SHAP Explainability")

    model_name = st.selectbox(
        "Model",
        list(trained_models.keys()),
        key="shap"
    )

    if st.button("Run SHAP"):

        model = trained_models.get(model_name)

        if model is None:
            st.warning("Train models first.")
        else:

            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            fig = plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(fig)


# =========================================================
# MODEL EXPORT
# =========================================================

st.sidebar.header("Export")

if st.sidebar.button("Download Best Model"):

    if best_model is None:
        st.sidebar.warning("Train models first.")
    else:

        with open("best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open("best_model.pkl", "rb") as f:
            st.sidebar.download_button(
                "Download",
                f,
                file_name="best_model.pkl"
            )
