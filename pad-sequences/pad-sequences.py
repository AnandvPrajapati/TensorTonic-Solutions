import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Determine target length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Create output filled with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copy each sequence (truncate if needed)
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        result[i, :len(trunc)] = trunc

    return result