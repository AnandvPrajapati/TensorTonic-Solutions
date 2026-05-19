import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T, dtype=float)
    points = np.array(points, dtype=float)

    # Check if input is a single point
    single_point = (points.ndim == 1)

    # Convert single point to batch format
    if single_point:
        points = points.reshape(1, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])

    # Apply transformation
    transformed_h = (T @ points_h.T).T

    # Extract spatial coordinates
    transformed = transformed_h[:, :3]

    # Return original shape
    if single_point:
        return transformed[0]

    return transformed