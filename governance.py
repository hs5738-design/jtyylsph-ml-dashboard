# governance.py

import numpy as np
from scipy.stats import wasserstein_distance


# =============================
# DRIFT
# =============================
def compute_drift(X_train, X_test):
    return wasserstein_distance(
        X_train.iloc[:, 0],
        X_test.iloc[:, 0]
    )


# =============================
# FAIRNESS
# =============================
def compute_fairness(preds, y_true):
    return abs(np.mean(preds) - np.mean(y_true))


# =============================
# STABILITY SCORE
# =============================
def system_stability_score(drift, fairness):
    return (1 - drift) * 0.5 + (1 - fairness) * 0.5


# =============================
# STATUS LABEL
# =============================
def status_label(value):
    if value < 0.3:
        return "🟢"
    elif value < 0.6:
        return "🟡"
    return "🔴"
