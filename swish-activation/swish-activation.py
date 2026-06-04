import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=np.float64)

    # Numerically stable sigmoid
    sigmoid = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

    return x * sigmoid