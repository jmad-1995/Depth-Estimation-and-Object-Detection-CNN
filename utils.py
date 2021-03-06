import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _gradient_loss(predictions, target):

    # Sobel X operator
    kernel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=target.dtype).view(1, 1, 3, 3)

    # Sobel Y operator
    kernel_y = torch.tensor([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=target.dtype).view(1, 1, 3, 3)

    # Gradient of predictions
    gx_p = F.conv2d(predictions, kernel_x.to(target.device), padding=1)
    gy_p = F.conv2d(predictions, kernel_y.to(target.device), padding=1)

    # Gradient of target
    gx_t = F.conv2d(target, kernel_x.to(target.device), padding=1)
    gy_t = F.conv2d(target, kernel_y.to(target.device), padding=1)

    gmag_p = torch.sqrt(gx_p**2 + gy_p**2)
    gmag_t = torch.sqrt(gx_t**2 + gy_t**2)

    return nn.MSELoss()(gmag_p, gmag_t)


def depth_loss(predictions, target):
    loss = 1. * nn.L1Loss()(predictions, target) + \
        _gradient_loss(predictions, target)
    return loss


def objects_loss(predictions, targets):
    return 0.


def depth_metrics(predictions, targets):

    """Numpy function to evaluate metrics of depth estimation"""

    if isinstance(predictions, list) or isinstance(targets, list):
        predictions = np.array(predictions)
        targets = np.array(targets)

    # Delta thresholds
    _max = np.maximum(predictions / targets, targets / predictions)
    deltas = []
    for thresh in [pow(1.25, 1), pow(1.25, 2), pow(1.25, 3)]:
        delta = np.count_nonzero(_max < thresh) / np.size(_max)
        deltas.append(delta)

    # Relative error
    rel = np.mean(np.abs(predictions - targets) / targets)

    # Root mean-squared error
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))

    # Log10 error
    log10 = np.mean(np.abs(np.log10(targets.clip(1e-3, None)) - np.log10(predictions.clip(1e-3, None))))

    return {'delta_1': deltas[0], 'delta_2': deltas[1], "delta_3": deltas[2], 'Rel': rel, 'RMSE': rmse, "log10": log10}


def objects_metrics(classes_prediction, bboxes_prediction, classes_targets, bboxes_targets):
    """Return mean Average precision given the class targets/predictions and bounding box target/predictions pairs for
       a single image.

    Arguments
    ---------
    classes_prediction: array (N, 1) of integers indicating the class the bounding box belongs to. 0=background, 1=car,
                        2=person.
    bboxes_prediction: array (N, 4) of bounding box coordinates [y1, x1, y2, x2] that correspond to the class
                       predictions.
    classes_targets: array (M, 1) of integers indicating the class the bounding box belongs to. 0=background, 1=car,
                     2=person.
    bboxes_prediction: array (M, 4) of bounding box coordinates [y1, x1, y2, x2] that correspond to the class
                       targets.

    Returns
    -------
    mAP: floating point value indicating the mean average precision for a set of detection targets and predictions from
         an image.

    """
    return


def get_object_depths(bboxes, classes, depth_map):

    depths = {'1': [], '2': []}
    for idx, bbox in enumerate(bboxes):

        # Bounding box in format [y1, x1, y2, x2]
        y1, x1, y2, x2 = np.array(bbox, np.int)

        # Get the average depth in the bounding box
        depth = depth_map[y1:y2, x1:x2].mean()
        if np.isnan(depth):
            continue
        else:
            # Append the result to the dictionary
            depths[str(classes[idx])].append(depth)

    return depths


def non_max_suppression(bboxes, min_IOU=.75):
    # bboxes are arrays with [x1,y1,x2,y2] co-ordinates for each bounding box that is predicted 
    # x1,y1 and x2,y2 are co-ordinates of top left and bottom right points of the bounding box
    
    # if bboxes is empty, return an empty list
    if len(bboxes) == 0: return []
        
    if bboxes.dtype.kind == "i": bboxes = bboxes.astype("float")

    # initialize the result list of selected bboxes
    result = []
    # divide coordinates of the bounding boxes
    x1,y1,x2,y2 = bboxes[:, 0],bboxes[:, 1],bboxes[:, 2],bboxes[:, 3]

    # compute each bbox area 
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort indices of x1 to compare with others.
    left_most_arg = np.argsort(x1)[::-1]

    # looping around x1 indices
    while len(left_most_arg) > 0:
        
        last = left_most_arg.shape[0] - 1
        k = left_most_arg[last]
        result.append(k)

        i_x1 = np.maximum(x1[k], x1[left_most_arg[:last]])
        i_y1 = np.maximum(y1[k], y1[left_most_arg[:last]])
        i_x2 = np.minimum(x2[k], x2[left_most_arg[:last]])
        i_y2 = np.minimum(y2[k], y2[left_most_arg[:last]])

        # Find intersection width and height
        width = np.maximum(0, i_x2 - i_x1 + 1)
        height = np.maximum(0, i_y2 - i_y1 + 1)

        #ratio of overlap
        IOU = (width * height) / area[left_most_arg[:last]]

        # remove overlaps
        left_most_arg = np.delete(left_most_arg, np.concatenate(([last],np.where(IOU > min_IOU)[0])))

    return bboxes[result]


