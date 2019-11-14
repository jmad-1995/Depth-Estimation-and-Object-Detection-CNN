import numpy as np


def generate_priors(image_shape, feature_map_shapes, prior_sizes, aspect_ratios):

    """Generate the numpy priors for the SSD network

    Arguments
    ---------
    image_shape: 2-element tuple (height, width) of the image shape
    feature_map_shapes: list of tuples [(height, width), (height, width)...] that correspond
                        to the feature map shapes.
    prior_sizes: list of integers corresponding to prior sizes for each feature map. '
                 len(prior_sizes) = len(feature_map_shapes)
    aspect_ratios: list of floating point values of corresponding to aspect ratios to use for priors

    Returns
    -------
    priors = array (N, 4) where N=number of priors flattened in the format (cy, cx, height, width)
    """

    priors = []
    for feature_map, size in zip(feature_map_shapes, prior_sizes):

        # Create a grid of all center x and center y values
        cx, cy = np.meshgrid(np.arange(feature_map[1]), np.arange(feature_map[0]))

        # Normalize
        cx = cx / feature_map[1]
        cy = cy / feature_map[0]

        # Generate the possible heights and widths of the priors and normalize
        heights = size * np.sqrt(aspect_ratios) / image_shape[0]
        widths = size / np.sqrt(aspect_ratios) / image_shape[1]

        # Create a prior for each height and width combo
        for h, w in zip(heights, widths):

            # Stack to create array (height, width, 4)
            prior_map = np.stack([cy, cx, np.ones_like(cy) * h, np.ones_like(cx) * w], axis=2)

            # Reshape
            priors.append(prior_map.reshape(-1, 4))

    return np.concatenate(priors, axis=0)


def generate_targets(bboxes, class_labels, priors, max_targets=100, min_negative_ratio=0.33):

    """Generate the targets by matching GT bounding boxes to the priors

    Arguments
    ---------
    bboxes: array (N, 4) of bounding boxes [y1, x2, y2, x2] (normalized coordinates) where N=number of GT detections
    class_labels: array (N, 1) of class labels where N=number of GT detections
    priors: array (P, 4) of the generated priors where P=total number of priors
    max_targets: maximum number of targets to use during training, set all others to -1
    min_negative_ratio: minimum ratio of negative:positive samples for hard negative mining

    Returns
    -------
    targets: dict {'class_targets', 'bbox_targets'}
    """

    # Reformat priors as [y1, x1, y2, x2]
    priors_bbox = np.zeros_like(priors)
    priors_bbox[:, 0] = priors[:, 0] - priors[:, 2] // 2
    priors_bbox[:, 1] = priors[:, 1] - priors[:, 3] // 2
    priors_bbox[:, 2] = priors[:, 0] + priors[:, 2] // 2
    priors_bbox[:, 3] = priors[:, 1] + priors[:, 3] // 2

    # Calculate the intersection over union overlap between GT bboxes and priors
    overlap = calculate_iou(bboxes, priors_bbox)

    # Find best prior for each bbox


def calculate_iou(bboxes, priors):

    # Create a grid of of all y1's, y2's, x1's, x2's
    y1_bboxes, y1_priors = np.meshgrid(bboxes[:, 0], priors[:, 0])
    x1_bboxes, x1_priors = np.meshgrid(bboxes[:, 1], priors[:, 1])
    y2_bboxes, y2_priors = np.meshgrid(bboxes[:, 2], priors[:, 2])
    x2_bboxes, x2_priors = np.meshgrid(bboxes[:, 3], priors[:, 3])

    # Find the rectangle that is inside each bbox and prior
    y1 = np.maximum(y1_bboxes, y1_priors)
    x1 = np.maximum(x1_bboxes, x1_priors)
    y2 = np.minimum(y2_bboxes, y2_priors)
    x2 = np.minimum(x2_bboxes, x2_priors)

    # Overlapping areas
    intersection = (y2 - y1) * (x2 - x1)

    # Union of areas
    boxes_area = (y2_bboxes - y1_bboxes) * (x2_bboxes - x1_bboxes)
    priors_area = (y2_priors - y1_priors) * (x2_priors - x1_priors)
    union = boxes_area + priors_area - intersection

    return intersection / union


def mold_offsets(bboxes, priors):
    """Convert normal form [cy, cx, h, w] into offsets [cy - by, cx - bx, log(bh/h), log(bw/w)]"""
    assert bboxes.shape == priors.shape
    offsets = np.zeros_like(bboxes)
    offsets[:, 0] = priors[:, 0] - bboxes[:, 0]
    offsets[:, 1] = priors[:, 1] - bboxes[:, 1]
    offsets[:, 2] = np.log(bboxes[:, 2] / priors[:, 2])
    offsets[:, 3] = np.log(bboxes[:, 3] / priors[:, 3])
    return offsets


def unmold_offsets(offsets, priors):
    """Convert offsets [cy - by, cx - cx, log(bh/h), log(bw/w)] into normal form [cy, cx, h, w]"""
    assert offsets.shape == priors.shape
    bboxes = np.zeros_like(offsets)
    bboxes[:, 0] = priors[:, 0] - offsets[:, 0]
    bboxes[:, 1] = priors[:, 1] - offsets[:, 1]
    bboxes[:, 2] = np.exp(offsets[:, 2]) * priors[:, 2]
    bboxes[:, 3] = np.exp(offsets[:, 3]) * priors[:, 3]
    return bboxes


def priors_to_bboxes(priors):
    """Convert [cy, cx, h, w] format to [y1, x1, y2, x2]"""
    bboxes = np.zeros_like(priors)
    bboxes[:, 0] = priors[:, 0] - priors[:, 2] // 2
    bboxes[:, 1] = priors[:, 1] - priors[:, 3] // 2
    bboxes[:, 2] = priors[:, 0] + priors[:, 2] // 2
    bboxes[:, 3] = priors[:, 1] + priors[:, 3] // 2
    return bboxes


def bboxes_to_priors(bboxes):
    """Convert [y1, x1, y2, x2] format to [cy, cx, h, w]"""
    priors = np.zeros_like(bboxes)
    priors[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    priors[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    priors[:, 0] = bboxes[:, 0] + priors[:, 2]
    priors[:, 1] = bboxes[:, 1] + priors[:, 3]
    return priors


if __name__ == '__main__':

    img_shape = (16, 16)
    fmaps = [(8, 8), (4, 4), (2, 2)]
    bs = [2, 4, 8]
    r = [1, .5, 2]

    p = generate_priors(img_shape, fmaps, bs, r)

    print(p)