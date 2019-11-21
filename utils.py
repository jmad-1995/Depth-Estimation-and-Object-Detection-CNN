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
    
    if len(bboxes) >= 0:
        x1 = bboxes[:,0]
        y1 = bboxes[;,1]
        x2 = bboxes[:,2]
        y2 = bboxes[;,3]
        
        chosen_bbox = []
       
        area = (x2 - x1) * (y2 - y1)
    else:
        return []
    
    
    return


def non_max_suppression_numpy(bboxes):
    return


