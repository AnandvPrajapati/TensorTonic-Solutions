import math
import numpy as np

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """
    feature_map = np.array(feature_map)
    pooled_outputs = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_h = y2 - y1
        roi_w = x2 - x1

        pooled = np.zeros((output_size, output_size))

        for i in range(output_size):
            for j in range(output_size):

                # Compute bin boundaries
                h_start = y1 + int(np.floor(i * roi_h / output_size))
                h_end = y1 + int(np.floor((i + 1) * roi_h / output_size))

                w_start = x1 + int(np.floor(j * roi_w / output_size))
                w_end = x1 + int(np.floor((j + 1) * roi_w / output_size))

                # Ensure at least one pixel
                if h_end == h_start:
                    h_end += 1

                if w_end == w_start:
                    w_end += 1

                # Extract region
                region = feature_map[h_start:h_end, w_start:w_end]

                # Max pooling
                pooled[i, j] = np.max(region)

        pooled_outputs.append(pooled.astype(int).tolist())

    return pooled_outputs