import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)


    Xt = X.T
    XtX = Xt @ X
    XtY = Xt @ y

    w = np.linalg.inv(XtX) @ XtY

    return w