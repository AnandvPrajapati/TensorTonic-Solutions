import numpy as np

def _sigmoid(z):
    # Numerically stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D, dtype=float)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # Gradients
        error = p - y
        dw = (X.T @ error) / N
        db = np.sum(error) / N

        # Parameter update
        w -= lr * dw
        b -= lr * db

    return w, float(b)