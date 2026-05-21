import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """


    v = np.array(v)

    D = np.diag(v)

    return D