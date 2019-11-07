import argparse
import torch

import networks
import utils
import kitti

# Argument parser
parser = argparse.ArgumentParser()
args = parser.parse_args()

# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Dataset
test_dataset = kitti.KITTIDataset(args.path, batch_size=args.batch_size)

# CNN
model = networks.MultiPurposeCNN().to(device)
model.load_state_dict(torch.load('models/{}'.format(args.saved_model)))
model.eval()


if __name__ == '__main__':

    metrics = {}
    for idx in range(len(test_dataset)):

        # Load data
        sample = test_dataset[idx]
        images = sample['images']
        depth_targets = sample['depths']
        object_targets = sample['objects']

        # Forward pass
        predictions = model(images.to(device))
        depth_predictions = predictions['depths']
        object_predictions = predictions['objects']

        # Convert to Numpy

        # Get metrics for both tasks
        metrics_objects = utils.objects_metrics(object_predictions, object_targets)
        metrics_depth = utils.depth_metrics(depth_predictions, depth_targets)

    # Export to CSV




