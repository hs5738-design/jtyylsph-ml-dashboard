import pandas as pd
import numpy as np
import time

def simulate_stream(X_test, steps=20, noise_level=0.05):
    current = X_test.copy().astype(float)

    for step in range(steps):
        try:
            noise = np.random.normal(0, noise_level, current.shape)
            current = current + noise
            current = current.fillna(0)  # ✅ prevent NaNs

            yield step, current

            time.sleep(0.2)

        except:
            yield step, current
