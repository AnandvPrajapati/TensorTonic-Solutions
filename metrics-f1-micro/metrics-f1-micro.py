import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Convert to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Positives = correct predictions
    tp = np.sum(y_true == y_pred)

    # Total samples
    total = len(y_true)

    # For single-label multi-class:
    # FP = FN = total - tp
    fp = total - tp
    fn = total - tp

    # Micro-F1 formula
    micro_f1 = (2 * tp) / (2 * tp + fp + fn)

    return float(micro_f1)