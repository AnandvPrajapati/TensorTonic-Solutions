import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    if seq_len < 1 or d_model < 1:
        raise ValueError("seq_len and d_model must both be >= 1")

    # Positions: shape (seq_len, 1)
    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]

    # Even dimension indices: 0, 2, 4, ...
    # Number of sin/cos frequency bands = ceil(d_model / 2)
    even_dims = np.arange(0, d_model, 2, dtype=float)

    # Compute angle rates: 1 / base^(2i / d_model)
    angle_rates = 1.0 / (base ** (even_dims / d_model))

    # Angles: shape (seq_len, ceil(d_model/2))
    angles = positions * angle_rates

    # Allocate output
    pe = np.empty((seq_len, d_model), dtype=float)

    # Fill even columns with sin
    pe[:, 0::2] = np.sin(angles)

    # Fill odd columns with cos (trim in case d_model is odd)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe