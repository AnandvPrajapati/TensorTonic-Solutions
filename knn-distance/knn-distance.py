import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Handle 1D input: shape (n,) -> (n,1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Edge case: no training points
    if n_train == 0:
        return np.full((n_test, k), -1, dtype=int)

    # Pairwise Euclidean distances using broadcasting
    # Shape: (n_test, n_train, d)
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))   # Shape: (n_test, n_train)

    # Sort indices by distance
    sorted_idx = np.argsort(dists, axis=1)

    # Take up to available neighbors
    take = min(k, n_train)
    result = np.full((n_test, k), -1, dtype=int)
    result[:, :take] = sorted_idx[:, :take]

    return result