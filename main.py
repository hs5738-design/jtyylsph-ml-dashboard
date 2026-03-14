def print_hi(name):
   print(f'Hi, {name}')


import matplotlib.pyplot as plt
from sklearn.metrics  import confusion_matrix, ConfusionMatrixDisplay
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import uuid, json, threading, joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
from api.predict import router as predict_router
from api.auth import router as auth_router

app = FastAPI(title="JTYYLSPH V7 API")

app.include_router(auth_router, prefix="/auth")
app.include_router(predict_router, prefix="/predict")
# -------------------------------
# 1. Core Model
# -------------------------------
class JTYYLSPHv7:
    def __init__(self, n_estimators=50, max_depth=3):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

        # ✅ Save FULL bundle correctly
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": {
                "accuracy": test_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "version": "v7"
        }

        joblib.dump(bundle, "jtyylsph_v46_bundle.pkl")
        print("Bundle saved correctly ✔")

        preds = self.model.predict(X_scaled)
        acc = accuracy_score(y, preds)

        return {"accuracy": acc}

    def predict(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X[self.feature_names])
        preds = self.model.predict(X_scaled)
        return {"predictions": preds.tolist()}


# -------------------------------
# 2. Mock SHAP
# -------------------------------
class MockSHAPExplainer:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def explain_instance(self, X_instance: pd.DataFrame):
        shap_values = np.random.randn(len(self.feature_names)) * 0.1
        return {
            "feature_importance": dict(zip(self.feature_names, np.abs(shap_values)))
        }


# -------------------------------
# 3. Logger
# -------------------------------
class SimpleLogger:
    def __init__(self, log_dir="logs_demo"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.pred_log = self.log_dir / "predictions.jsonl"

    def log_prediction(self, data: dict):
        with open(self.pred_log, "a") as f:
            entry = {"_id": str(uuid.uuid4()), "_timestamp": datetime.now().isoformat(), "_source": data}
            f.write(json.dumps(entry) + "\n")


# -------------------------------
# 4. Scheduler
# -------------------------------
class PersistentRetrainingScheduler:
    def __init__(self, model, check_interval=3):
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)

    def _run(self):
        while not self._stop_event.is_set():
            print("[Scheduler] Checking model retraining needs...")
            self._stop_event.wait(3)

    def start(self): self.thread.start()
    def stop(self):
        self._stop_event.set()
        self.thread.join()


# -------------------------------
# 5. Enhanced Model
# -------------------------------
class EnhancedJTYYLSPHv44(JTYYLSPHv44):
    def __init__(self):
        super().__init__()
        self.logger = SimpleLogger()
        self.shap_explainer = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True

        # Compute metrics on training data
        preds = self.model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        # For simplicity, precision/recall/F1 can be left None or computed if y has positive class
        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": {"accuracy": acc, "precision": None, "recall": None, "f1": None},
            "version": "v4.6"
        }
        joblib.dump(bundle, "jtyylsph_v46_bundle.pkl")
        print("Bundle saved correctly ✔")
        return {"accuracy": acc}

    def explain(self, X_instance):
        return self.shap_explainer.explain_instance(X_instance)


# -------------------------------
# 6. DEMO PIPELINE
# -------------------------------
async def run_demo():
    print("=" * 50)
    print("JTYYLSPH v4.5 Extended Demo")
    print("=" * 50)

    model = EnhancedJTYYLSPHv44()
    print("[1] Model initialized.")

    # Generate full dataset
    X_full = pd.DataFrame({
        "feature_1": np.random.randn(300),
        "feature_2": np.random.randn(300),
        "volatility": np.random.rand(300)
    })
    y_full = ((X_full["feature_1"] > 0) &
              (X_full["feature_2"] > 0)).astype(int)

    # Proper train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42
    )

    print("[2] Running GridSearch...")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [2, 3, 4]
    }

    grid = GridSearchCV(
        GradientBoostingClassifier(),
        param_grid,
        cv=3
    )

    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)

    # Use best model
    model.model = grid.best_estimator_

    print("[3] Training best model...")
    result = model.train(X_train, y_train)
    print(f"Accuracy (Train): {result['accuracy']:.4f}")

    # Evaluate on TEST data
    X_test_scaled = model.scaler.transform(X_test)
    y_pred_test = model.model.predict(X_test_scaled)

    from sklearn.metrics import precision_score, recall_score, f1_score

    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    print(f"Accuracy (Test): {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 📊 Confusion Matrix (Test Set)
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix (Test Data)")
    plt.show()

    # 📈 Feature Importance
    importances = model.model.feature_importances_
    features = model.feature_names

    plt.figure()
    plt.bar(features, importances)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    scheduler = PersistentRetrainingScheduler(model)
    scheduler.start()
    print("[4] Scheduler running...")
    await asyncio.sleep(6)
    scheduler.stop()
    print("[4] Scheduler stopped.")

# -------------------------------
# RUN EVERYTHING
# -------------------------------
if __name__ == "__main__":
    print_hi("PyCharm")
    print("Everything works")
    asyncio.run(run_demo())
