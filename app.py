import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import datetime
import json
from sklearn.datasets import make_classification

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="JTYYLSPH AI Dashboard", layout="wide")
st.title("📊 JTYYLSPH AI Dashboard - Polished Tabs Layout")

# -------------------------------
# Load or Generate Demo Data
# -------------------------------
X, y = make_classification(
    n_samples=300,
    n_features=3,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=42
)

X = pd.DataFrame(X, columns=["feature_1", "feature_2", "volatility"])
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Models
# -------------------------------
models_dict = {
    "GradientBoosting": GradientBoostingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

# Load pretrained GradientBoosting bundle
bundle_path = "jtyylsph_v46_bundle.pkl"
bundle = joblib.load(bundle_path)
gb_model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
models_dict["GradientBoosting"] = gb_model

# -------------------------------
# Prediction Logger
# -------------------------------
def log_prediction(input_data, prediction, probability=None, model_name="Unknown"):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": input_data,
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else None,
        "model": model_name
    }
    with open("prediction_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Manual Prediction", "Batch Prediction", "Feature Importance", "EDA"])

# -------------------------------
# Tab 1: Manual Prediction
# -------------------------------
with tabs[0]:
    st.header("🖊 Manual Prediction")
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    vol = st.number_input("Volatility", value=0.5)
    selected_model_name = st.selectbox("Select Model", list(models_dict.keys()))

    if st.button("Predict", key="manual_predict"):
        input_df = pd.DataFrame([[f1, f2, vol]], columns=feature_names)
        try:
            model = models_dict[selected_model_name]
            X_input = scaler.transform(input_df) if selected_model_name == "GradientBoosting" else input_df.values
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

            st.subheader("Prediction Result")
            st.write(f"Model: {selected_model_name}")
            st.write("Class:", prediction)
            if probability is not None:
                st.write(f"Prediction Confidence: {probability:.2f}")

            log_prediction(input_df.to_dict(), prediction, probability, selected_model_name)

            if probability is not None:
                fig, ax = plt.subplots()
                ax.barh(["Probability"], [probability])
                ax.set_xlim(0, 1)
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# Tab 2: Batch Prediction & Comparison
# -------------------------------
with tabs[1]:
    st.header("📂 Batch Prediction & Model Comparison")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_compare")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            missing_cols = [c for c in feature_names if c not in data.columns]
            if missing_cols:
                st.error(f"Uploaded CSV missing required features: {', '.join(missing_cols)}")
            else:
                st.subheader("Predictions & Accuracy by Model")
                comparison_results = {}

                for name, model in models_dict.items():
                    X_scaled = scaler.transform(data[feature_names]) if name == "GradientBoosting" else data[feature_names].values
                    preds = model.predict(X_scaled)
                    data[f"Prediction_{name}"] = preds

                    if "target" in data.columns:
                        acc = accuracy_score(data["target"], preds)
                        comparison_results[name] = acc

                        st.write(f"Confusion Matrix: {name}")
                        cm = confusion_matrix(data["target"], preds)
                        fig_cm, ax_cm = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                        disp.plot(ax=ax_cm)
                        st.pyplot(fig_cm)

                # Accuracy Ranking Bar Chart
                if comparison_results:
                    st.subheader("📊 Model Accuracy Ranking")
                    acc_df = pd.DataFrame(list(comparison_results.items()), columns=["Model", "Accuracy"])
                    acc_df = acc_df.sort_values("Accuracy", ascending=False)
                    st.write(acc_df)

                    # Highlight Best Model
                    best_model = acc_df.iloc[0]
                    st.markdown(f"### 🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.2f})")

                    fig_acc, ax_acc = plt.subplots()
                    ax_acc.barh(acc_df["Model"], acc_df["Accuracy"], color='skyblue')
                    ax_acc.set_xlim(0, 1)
                    ax_acc.set_xlabel("Accuracy")
                    ax_acc.set_title("Model Accuracy Ranking")
                    for i, v in enumerate(acc_df["Accuracy"]):
                        ax_acc.text(v + 0.01, i, f"{v:.2f}", va='center')
                    st.pyplot(fig_acc)

                # Download predictions
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button("Download All Model Predictions", csv, "predictions_comparison.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# -------------------------------
# Tab 3: Feature Importance
# -------------------------------
with tabs[2]:
    st.header("📊 Feature Importance")
    for name, model in models_dict.items():
        if name in ["GradientBoosting", "RandomForest"] and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig, ax = plt.subplots()
            ax.bar(feature_names, importances)
            ax.set_title(f"{name} Feature Importance")
            st.pyplot(fig)
        else:
            st.write(f"Feature importance not available for {name}.")

# -------------------------------
# Tab 4: Exploratory Data Analysis (EDA)
# -------------------------------
with tabs[3]:
    st.header("📊 Exploratory Data Analysis")
    uploaded_file_eda = st.file_uploader("Upload CSV for EDA", type=["csv"], key="eda")
    if uploaded_file_eda:
        try:
            df_eda = pd.read_csv(uploaded_file_eda)
            st.subheader("Dataset Preview")
            st.write(df_eda.head())
            st.subheader("Summary Statistics")
            st.write(df_eda.describe())
            st.subheader("Correlation Matrix")
            corr = df_eda.corr()
            fig_corr, ax_corr = plt.subplots()
            cax = ax_corr.matshow(corr, cmap="coolwarm")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.colorbar(cax)
            st.pyplot(fig_corr)
        except Exception as e:
            st.error(f"Error during EDA: {e}")
