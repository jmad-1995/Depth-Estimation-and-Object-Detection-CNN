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

    return bboxes[result].astype("int")


def non_max_suppression_numpy(bboxes):
    return


