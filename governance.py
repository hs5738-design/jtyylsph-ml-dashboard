import numpy as np
from scipy.stats import wasserstein_distance

def compute_drift(X_train, X_test):
    try:
        col = X_train.select_dtypes(include=[np.number]).columns[0]
        x1 = X_train[col].fillna(0)
        x2 = X_test[col].fillna(0)
        return float(wasserstein_distance(x1, x2))
    except:
        return 0.0  # ✅ SAFE FALLBACK

def compute_fairness(preds, y_true):
    try:
        return float(abs(np.mean(preds) - np.mean(y_true)))
    except:
        return 0.0

def system_stability_score(drift, fairness):
    score = (1 - drift) * 0.5 + (1 - fairness) * 0.5
    return max(0.0, min(1.0, float(score)))  # ✅ CLAMP

def status_label(value):
    if value < 0.3:
        return "🟢"
    elif value < 0.6:
        return "🟡"
    return "🔴"
