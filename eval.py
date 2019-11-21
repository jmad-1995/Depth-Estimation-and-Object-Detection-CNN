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
test_dataset = kitti.KITTI(args.path, batch_size=args.batch_size)

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
        predictions = model(torch.stack([images, torch.flip(images, dims=[3])]).to(device))
        depth_predictions = torch.mean(torch.stack([predictions['depths'][0], torch.flip(predictions['depth'][1], dims=[2])]))

        # Convert to Numpy

        # Get metrics for depth task
        metrics_depth = utils.depth_metrics(depth_predictions, depth_targets)

        # Get metrics for specific objects: cars, people

    # Export to CSV




