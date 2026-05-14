import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Convert input to NumPy array
    x = np.array(x, dtype=float)

    # Random values
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    # Keep mask (True = keep neuron)
    keep_mask = random_vals >= p

    # Inverted dropout scaling
    scale = 1.0 / (1.0 - p)

    # Output after dropout
    output = x * keep_mask * scale

    # Dropout pattern (scaled mask)
    dropout_pattern = keep_mask.astype(float) * scale

    return output, dropout_pattern