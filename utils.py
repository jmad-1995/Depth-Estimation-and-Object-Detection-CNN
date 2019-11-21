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
    loss = nn.L1Loss()(predictions, target) + \
           _gradient_loss(predictions, target)
    return loss


def objects_loss(predictions, targets):
    return


def depth_metrics(predictions, targets):

    """Numpy function to evaluate metrics of depth estimation"""

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
    log10 = np.mean(np.abs(np.log10(targets) - np.log10(predictions)))

    return {'delta_1': deltas[0], 'delta_2': deltas[1], "delta_3": deltas[2], 'Rel': rel, 'RMSE': rmse, "log10": log10}


def objects_metrics(classes_prediction, bboxes_prediction, classes_targets, bboxes_targets):
    """Return mean Average precision given the class targets/predictions and bounding box target/predictions pairs

    Arguments
    ---------

    """
    return


def non_max_suppression(bboxes, min_IOU=.75):
    return


def non_max_suppression_numpy(bboxes):
    return


