import torch
import torch.nn as nn
import numpy as np


def depth_loss(predictions, targets):
    return


def objects_loss(predictions, targets):
    return


def depth_metrics(predictions, targets):
    return


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


