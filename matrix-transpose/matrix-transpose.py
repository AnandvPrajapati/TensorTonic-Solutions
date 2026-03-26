import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    N, M = A.shape

    # Create output matrix with swapped shape
    result = np.empty((M, N), dtype=A.dtype)

    # Manually swap indices
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]

    return result