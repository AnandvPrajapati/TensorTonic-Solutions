import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x, dtype = float)

    mean = np.mean(x)
    
    median = np.median(x)

    freq = Counter(x)
    max_freq = max(freq.values())

    mode = min(num for num, count in freq.items() if count == max_freq)

    return float(mean), float(median), float(mode)