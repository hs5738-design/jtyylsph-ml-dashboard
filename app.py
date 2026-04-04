Establishing rules and timing is like setting up a formation for the Qimen Dunjia, and creating a holographic AI linked to the Earth system to simulate the operation status and laws of celestial bodies
# =============================
# 🚀 JTYYLSPH — AI Governance Platform
# =============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import time
# =============================
# CUSTOM MODULES
# =============================
try:
    from core import simulate_stream
    from report import generate_pdf_report
    from governance import compute_drift, compute_fairness, system_stability_score, status_label
    from audit import log_run, load_logs
except ModuleNotFoundError as e:
    st.error(f"Missing module: {e}")
    st.stop()
# =============================
# FUNCTIONS
# =============================
def ingest_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif file.name.endswith(".json"):
            return pd.read_json(file)
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to read {file.name}: {e}")
        return None
# =============================
# DARK MODE TOGGLE
# =============================
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
# =============================
# PAGE TITLE
# =============================
st.title("🚀 JTYYLSPH — AI Governance Platform")
# =============================
# SESSION STATE
# =============================
if "model" not in st.session_state:
    st.session_state.model = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# =============================
# SIDEBAR
# =============================
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
st.sidebar.header("Dataset Controls")
domain = st.sidebar.selectbox(
    "Synthetic Dataset",
    ["Finance", "Healthcare", "Sports", "Business", "Emotion", "General"]
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
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
st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input("SQLAlchemy DB URL", placeholder="postgresql://user:pass@host:5432/db")
query = st.sidebar.text_area("SQL Query", placeholder="SELECT * FROM table LIMIT 100")
# =============================
# DATA LOADING
# =============================
X, y, df = None, None, None
if query and db_url:
    from sqlalchemy import create_engine
    try:
        engine = create_engine(db_url)
        df = pd.read_sql(query, engine)
        st.write("Database Data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()
elif uploaded_files:
    dfs = []
    for f in uploaded_files:
        df_part = ingest_file(f)
        if df_part is not None:
            dfs.append(df_part)
    if dfs:
        df = pd.concat(dfs, ignore_index=True, sort=False)
        st.write("Combined Dataset")
        st.dataframe(df.head())
elif uploaded:
    df = pd.read_csv(uploaded)
    st.write("Uploaded Dataset")
    st.dataframe(df.head())
else:
    X_data, y_data = make_classification(n_samples=500, n_features=6, random_state=42)
    df = pd.DataFrame(X_data)
    df["target"] = y_data
    st.info(f"Using synthetic dataset: {domain}")
# =============================
# TARGET SELECTION
# =============================
if df is not None:
    if len(df.columns) > 1:
        target_col = st.sidebar.selectbox("Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = pd.Series(np.random.randint(0, 2, len(df)))
if X is None or len(X) == 0:
    st.error("No valid dataset loaded.")
    st.stop()
X.columns = [str(c) for c in X.columns]
st.write("Dataset Shape:", X.shape)
st.write("### Dataset Summary")
st.write(X.describe())
# =============================
# TARGET VALIDATION
# =============================
if y.isna().any():
    st.error("Target contains missing values.")
    st.stop()
unique_vals = pd.unique(y)
is_classification = y.dtype == "object" or str(y.dtype).startswith("category") or len(unique_vals) <= 20
if is_classification:
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y)
else:
    st.warning("Regression detected — switching model")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X, y)
    st.success("Regression model trained")
    st.stop()
# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# =============================
# DATA QUALITY
# =============================
st.subheader("📊 Data Quality Check")
st.write("Missing Values")
st.write(X.isna().sum())
st.write("Duplicate Rows")
st.write(X.duplicated().sum())
# =============================
# MODEL SELECTION & TRAINING
# =============================
model_choice = st.selectbox("Select Model", ["RandomForest", "GradientBoosting", "LogisticRegression"])
st.write("Target dtype:", y.dtype)
st.write("Unique target values:", np.unique(y)[:20])
st.write("Number of unique classes:", len(np.unique(y)))
if st.button("Train Model"):
    if model_choice == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif model_choice == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    drift = compute_drift(X_train, X_test)
    fairness = compute_fairness(preds, y_test)
    stability = system_stability_score(drift, fairness)
    st.session_state.model = model
    st.session_state.metrics = (drift, fairness, stability)
    log_run(model_choice, drift, fairness, stability, jurisdiction)
# =============================
# RISK DASHBOARD
# =============================
if st.session_state.metrics:
    drift, fairness, stability = st.session_state.metrics
    st.subheader("📊 Risk Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift", round(drift, 3), status_label(drift))
    c2.metric("Fairness", round(fairness, 3), status_label(fairness))
    c3.metric("Stability", round(stability, 3), status_label(1 - stability))
    if st.button("📄 Generate Report"):
        file_path = generate_pdf_report(drift, fairness, stability)
        with open(file_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="AI_Risk_Report.pdf", mime="application/pdf")
    st.subheader("🧾 Interpretation")
    if drift > 0.3: st.warning("Drift detected — retrain model")
    if fairness > 0.1: st.warning("Bias risk detected")
    if stability < 0.5: st.error("🔴 HIGH RISK SYSTEM")
    else: st.success("System stable")
# =============================
# REAL-TIME SIMULATION
# =============================
st.subheader("🧠 Real-Time Simulation")
if st.session_state.model and st.button("Start Simulation"):
    chart = st.line_chart()
    for step, current_data in simulate_stream(X_test):
        preds = st.session_state.model.predict(current_data)
        drift = compute_drift(X_train, current_data)
        fairness = compute_fairness(preds, y_test)
        chart.add_rows({"Drift": [drift], "Fairness": [fairness]})
# =============================
# AUDIT LOG
# =============================
st.subheader("📜 Audit Log")
logs = load_logs()
if logs:
    st.dataframe(pd.DataFrame(logs))
else:
    st.info("No audit logs yet.")
# =============================
# AI GOVERNANCE ASSISTANT
# =============================
st.subheader("🤖 AI Governance Assistant")

import os
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    st.warning("⚠️ OpenAI API key not set. AI assistant disabled.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# Quick buttons
col1, col2, col3 = st.columns(3)
quick_prompt = None
if col1.button("Explain Risk"): quick_prompt = "Explain current system risk"
if col2.button("Is model biased?"): quick_prompt = "Is this model biased?"
if col3.button("Should I retrain?"): quick_prompt = "Should I retrain the model?"

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
chat_input = st.chat_input("Ask about your model...")
user_input = quick_prompt if quick_prompt else chat_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build system context SAFELY
    system_context = "You are a senior AI risk officer."

    if st.session_state.metrics:
        drift, fairness, stability = st.session_state.metrics
        system_context += f"""
        Current system metrics:
        - Drift: {round(drift,3)}
        - Fairness: {round(fairness,3)}
        - Stability: {round(stability,3)}
        Dataset shape: {X.shape}
        """

    # Call OpenAI ONLY if client exists
    if client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_context}] + st.session_state.messages
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ AI error: {e}"
    else:
        reply = "⚠️ AI assistant unavailable (no API key)."

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Streaming effect
    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed = ""
        for char in reply:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.01)

