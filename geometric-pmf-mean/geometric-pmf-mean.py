import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.array(k, dtype=int)

    # Compute PMF
    pmf = ((1 - p) ** (k - 1)) * p

    # Compute mean
    mean = 1 / p

    return pmf, float(mean)