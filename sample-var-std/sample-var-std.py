import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.array(x, dtype=float)

    # Sample variance with Bessel correction (ddof=1)
    var = np.var(x, ddof=1)

    # Sample standard deviation
    std = np.std(x, ddof=1)

    return float(var), float(std)