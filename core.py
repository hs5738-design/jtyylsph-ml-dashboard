# core.py

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def prepare_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = X.select_dtypes(include=[np.number]).fillna(0)

    if X.shape[1] == 0:
        raise ValueError("No numeric features found.")

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)


# 🔥 Simulation (fixed + safer)
def simulate_stream(X_test, steps=20, noise_level=0.05):
    current = X_test.copy()

    for step in range(steps):
        noise = noise_level * current.std() * np.random.randn(*current.shape)
        current = current + noise

        yield step, current
        time.sleep(0.2)
