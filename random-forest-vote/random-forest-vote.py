import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
   
    preds = np.array(predictions)   # shape: (T, N)
    T, N = preds.shape
    result = []

    for i in range(N):
        votes = preds[:, i]
        counts = np.bincount(votes)
        result.append(np.argmax(counts))

    return result