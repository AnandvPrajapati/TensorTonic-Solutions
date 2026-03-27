import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """

    x = np.array(x, dtype=np.float64)
    gamma = np.array(gamma, dtype=np.float64)
    beta = np.array(beta, dtype=np.float64)
    if x.ndim == 2:
        mean = np.mean(x, axis = 0, keepdims = True)
        var = np.var(x, axis = 0, keepdims = True)

        x_hat = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_hat + beta

    elif x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
        var  = np.var(x, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)

        x_hat = (x - mean) / np.sqrt(var + eps)           # (N, C, H, W)

        gamma = gamma.reshape(1, -1, 1, 1)               # (1, C, 1, 1)
        beta  = beta.reshape(1, -1, 1, 1)                # (1, C, 1, 1)

        y = gamma * x_hat + beta

    return y
        