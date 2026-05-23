import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    n = len(y)

    # Empty node
    if n == 0:
        return 0.0

    # Count class occurrences
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    probs = counts / n

    # Stable entropy computation
    # Ignore zero probabilities to avoid log2(0)
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)