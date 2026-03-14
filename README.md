## 📜 License

MIT License
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)


# JTYYLSPH — Enterprise AI Governance Platform

An end-to-end machine learning system for predictive classification, explainability, and decision analytics across multiple domains including finance, healthcare, and risk modeling.

This project demonstrates production-oriented ML engineering with model evaluation, visualization, logging, and deployment via an interactive web application.

---

## 🚀 Live Demo

Streamlit App:  
https://hs5738-design-jtyylsph-ml-dashboard-app-3ahbbg.streamlit.app/

---

## 📂 Repository

GitHub:  
https://github.com/hs5738-design/jtyylsph-ml-dashboard

---

## 🎯 Key Features


- AutoML model training
- Model registry
- Bias detection
- Drift monitoring
- Model approval workflow
- REST API
- LLM explainability
- Streamlit dashboard
- End-to-end machine learning pipeline
- Model training, evaluation, and comparison
- Precision, Recall, F1-score, ROC curve analysis
- Confusion matrix visualization
- Feature importance & correlation analysis
- Dataset upload for experimentation
- Prediction logging and model versioning
- Explainability via feature attribution
- Interactive Streamlit dashboard
- Modular architecture for multi-domain use

---

## 🧠 Architecture

Frontend:
Streamlit dashboard

Backend:
FastAPI service

Storage:
SQL model registry

Monitoring:
Drift detection + bias monitoring

## Run locally

pip install -r requirements.txt

uvicorn api.main:app --reload

streamlit run dashboard/streamlit_app.py

---

## 🛠️ Technology Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Streamlit
- Joblib
- FastAPI (architecture design)
- Docker (deployment-ready design)
- Redis (caching architecture concept)

---

## 📊 Model Capabilities

The platform supports:

- Classification tasks
- Risk prediction
- Decision analytics
- Multi-domain datasets

Example performance:

| Domain | Accuracy |
|--------|----------|
Financial Risk | 88% |
Synthetic Classification | 90%+ |

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve & AUC
- Confusion Matrix

---

## 📊 Visual Analytics

The dashboard includes:

- Confusion Matrix
- Feature Importance
- Feature Correlation Heatmap
- ROC Curves
- Exploratory Data Analysis

---

## 📁 Project Structure



JTYYLSPH-V7/
│
├── api/
│   ├── main.py
│   ├── auth.py
│   ├── predict.py
│   ├── models.py
│   └── schemas.py
│
├── automl/
│   ├── automl_engine.py
│   └── trainer.py
│
├── explainability/
│   ├── shap_engine.py
│   └── llm_explainer.py
│
├── monitoring/
│   ├── drift.py
│   ├── psi.py
│   └── alerts.py
│
├── governance/
│   ├── workflow.py
│   ├── approvals.py
│   └── risk_rating.py
│
├── registry/
│   ├── sql_registry.py
│   └── migrations.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_api.py
│   └── test_models.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── scripts/
│   ├── train_model.py
│   └── deploy_model.py
│
├── models/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .github/
    └── workflows/
        └── ci.yml


---

## ⚙️ Installation

Clone the repository:



git clone https://github.com/hs5738-design/jtyylsph-ml-dashboard.git

cd jtyylsph-ml-dashboard


Install dependencies:



pip install -r requirements.txt


Run the application:



streamlit run app.py


---

## 🔬 Example Use Cases

- Financial risk prediction
- Healthcare classification
- Policy or governance analytics
- Supply chain risk modeling
- Experimental ML research

---

## 📌 Engineering Highlights

- Modular and extensible architecture
- Explainable ML design principles
- Production-oriented logging and versioning
- Multi-domain adaptability
- Interactive visualization interface
- Reproducible training pipeline

---

## 👨‍💻 Author

Kaleb Carter Shi

---
