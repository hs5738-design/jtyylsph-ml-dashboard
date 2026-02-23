import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="ML Dashboard V7 ELITE", layout="wide")

MODEL_DIR = "models"
REGISTRY_FILE = "registry.json"

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Registry Utilities
# -----------------------------
def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return []

    try:
        with open(REGISTRY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except:
        return []


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(name, path, accuracy):
    registry = load_registry()

    record = {
        "name": name,
        "path": path,
        "accuracy": float(accuracy)
    }

    registry.append(record)
    save_registry(registry)


# -----------------------------
# Load Models Safely
# -----------------------------
def load_models():
    registry = load_registry()
    models = []

    for rec in registry:
        try:
            path = rec.get("path")

            if not path or not os.path.exists(path):
                continue

            model = joblib.load(path)

            models.append({
                "name": rec.get("name", "Unknown"),
                "model": model,
                "accuracy": rec.get("accuracy", 0)
            })

        except Exception as e:
            st.warning(f"Failed loading model: {rec}")

    return models


# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Train Model", "Comparison", "Monitoring"]
)

st.title("🚀 ML Dashboard V7 ELITE")


# -----------------------------
# Upload Data
# -----------------------------
if menu == "Upload Data":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state["data"] = df
        st.success("Data loaded")
        st.dataframe(df.head())


# -----------------------------
# Train Model
# -----------------------------
elif menu == "Train Model":

    if "data" not in st.session_state:
        st.warning("Upload data first")
        st.stop()

    df = st.session_state["data"]

    target = st.selectbox("Select Target Column", df.columns)

    model_type = st.selectbox(
        "Model Type",
        ["RandomForest", "LogisticRegression"]
    )

    if st.button("Train"):

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_type == "RandomForest":
            model = RandomForestClassifier()
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        model_name = f"{model_type}_{len(load_registry())+1}"
        model_path = os.path.join(MODEL_DIR, model_name + ".pkl")

        joblib.dump(model, model_path)

        register_model(model_name, model_path, acc)

        st.success(f"Model trained: {model_name}")
        st.write("Accuracy:", acc)


# -----------------------------
# Comparison
# -----------------------------
elif menu == "Comparison":

    models = load_models()

    if not models:
        st.warning("No models available")
        st.stop()

    df = pd.DataFrame([
        {"Model": m["name"], "Accuracy": m["accuracy"]}
        for m in models
    ])

    st.subheader("Model Comparison")
    st.dataframe(df)

    st.bar_chart(df.set_index("Model"))


# -----------------------------
# Monitoring
# -----------------------------
elif menu == "Monitoring":

    models = load_models()

    if not models:
        st.warning("No models available")
        st.stop()

    best_model = max(models, key=lambda x: x["accuracy"])

    st.subheader("Best Model")
    st.write("Name:", best_model["name"])
    st.write("Accuracy:", best_model["accuracy"])

    st.success("System Healthy ✅")
