import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    def gini(y):
        n = len(y)

        # Empty node => impurity 0
        if n == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / n

        return 1.0 - np.sum(probs ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    # Avoid division by zero
    if n_total == 0:
        return 0.0

    return (
        (n_left / n_total) * gini(y_left) +
        (n_right / n_total) * gini(y_right)
    )