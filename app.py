
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from core import simulate_stream
from report import generate_pdf_report
from governance import (
    compute_drift,
    compute_fairness,
    system_stability_score,
    status_label
)
from audit import log_run, load_logs

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
query = st.sidebar.text_area("SQL Query")

# =============================
# SIDEBAR INPUTS (DEFINE FIRST)
# =============================
# DATABASE
st.sidebar.header("Database Connection")
db_url = st.sidebar.text_input(
    "SQLAlchemy DB URL",
    placeholder="postgresql://user:pass@host:5432/db"
)
query = st.sidebar.text_area(
    "SQL Query",
    placeholder="SELECT * FROM table LIMIT 100"
)

# =============================
# DATA SOURCE PRIORITY SYSTEM
# =============================
X, y = None, None
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
    X_data, y_data = make_classification(
        n_samples=500, n_features=6, random_state=42
    )
    df = pd.DataFrame(X_data)
    df["target"] = y_data
    st.info(f"Using synthetic dataset: {domain}")
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
    except:
        return None

# =============================
# TARGET SELECTION (UNIFIED)
# =============================
if 'df' in locals():
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
# TARGET VALIDATION (MOVE HERE)
# =============================
from sklearn.preprocessing import LabelEncoder
if y.isna().any():
    st.error("Target contains missing values.")
    st.stop()
unique_vals = pd.unique(y)
is_classification = (
    y.dtype == "object"
    or str(y.dtype).startswith("category")
    or len(unique_vals) <= 20
)
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
from sklearn.model_selection import train_test_split

X.columns = [str(c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42
)

# =============================
# DATA QUALITY
# =============================
st.subheader("📊 Data Quality Check")

st.write("Missing Values")
st.write(X.isna().sum())

st.write("Duplicate Rows")
st.write(X.duplicated().sum())

# =============================
# MODEL SELECTION
# =============================
model_choice = st.selectbox(
    "Select Model",
    ["RandomForest", "GradientBoosting", "LogisticRegression"]
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# =============================
# TRAIN
# =============================
st.write("Target dtype:", y.dtype)
st.write("Unique target values:", y.unique()[:20])
st.write("Number of unique classes:", len(pd.unique(y)))

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
    # ✅ FIXED: log inside training
    log_run(
        model_choice,
        drift,
        fairness,
        stability,
        jurisdiction
    )

# =============================
# SHOW RESULTS (PERSISTENT)
# =============================
if st.session_state.metrics:

    drift, fairness, stability = st.session_state.metrics

    st.subheader("📊 Risk Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Drift", round(drift, 3), status_label(drift))
    c2.metric("Fairness", round(fairness, 3), status_label(fairness))
    c3.metric("Stability", round(stability, 3), status_label(1 - stability))

    # =============================
    # PDF REPORT
    # =============================
    if st.button("📄 Generate Report"):
        file_path = generate_pdf_report(drift, fairness, stability)

        with open(file_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="AI_Risk_Report.pdf",
                mime="application/pdf"
            )

    # =============================
    # INTERPRETATION
    # =============================
    st.subheader("🧾 Interpretation")

    if drift > 0.3:
        st.warning("Drift detected — retrain model")

    if fairness > 0.1:
        st.warning("Bias risk detected")

    if stability < 0.5:
        st.error("🔴 HIGH RISK SYSTEM")
    else:
        st.success("System stable")
  
    # =============================
    # REAL-TIME SIMULATION
    # =============================
    st.subheader("🧠 Real-Time Simulation")

    if st.button("Start Simulation"):

        chart = st.line_chart()

        model = st.session_state.model

        for step, current_data in simulate_stream(X_test):

            preds = model.predict(current_data)

            drift = compute_drift(X_train, current_data)
            fairness = compute_fairness(preds, y_test)

            chart.add_rows({
                "Drift": [drift],
                "Fairness": [fairness]
            })
st.subheader("📜 Audit Log")
logs = load_logs()
if logs:
    df_logs = pd.DataFrame(logs)
    st.dataframe(df_logs)
else:
    st.info("No audit logs yet.")
if X is None or len(X) == 0:
    st.error("No valid dataset loaded.")
    st.stop()
X.columns = [str(c) for c in X.columns]
st.write("Dataset Shape:", X.shape)
st.write("### Dataset Summary")
st.write(X.describe())
# =============================
# 🤖 AI ASSISTANT (WORKING)
# =============================
from openai import OpenAI
import time
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.subheader("🤖 AI Governance Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []
# Quick buttons
col1, col2, col3 = st.columns(3)
quick_prompt = None
if col1.button("Explain Risk"):
    quick_prompt = "Explain current system risk"
if col2.button("Is model biased?"):
    quick_prompt = "Is this model biased?"
if col3.button("Should I retrain?"):
    quick_prompt = "Should I retrain the model?"
# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
# Input
chat_input = st.chat_input("Ask about your model...")
user_input = quick_prompt if quick_prompt else chat_input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Build system context
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
    else:
        system_context += "\nNo model trained yet. Help user set up model."
    # Call AI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_context}
            ] + st.session_state.messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ AI error: {str(e)}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # Typing effect
    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed = ""
        for char in reply:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.01)


