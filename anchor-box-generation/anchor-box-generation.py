import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """

    anchors = []

    # Step 1: Compute stride
    stride = image_size / feature_size

    # Step 2: Iterate over feature grid
    for i in range(feature_size):
        for j in range(feature_size):

            # Grid cell center in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # Step 3: Generate anchors
            for s in scales:
                for r in aspect_ratios:

                    # Width and height
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)

                    # Anchor coordinates
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ])

    return anchors