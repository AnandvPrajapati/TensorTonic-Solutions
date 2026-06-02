import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Create word -> index mapping
    vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Initialize count vector
    bow = np.zeros(len(vocab), dtype=int)

    # Count occurrences
    for token in tokens:
        if token in vocab_to_idx:
            bow[vocab_to_idx[token]] += 1

    return bow