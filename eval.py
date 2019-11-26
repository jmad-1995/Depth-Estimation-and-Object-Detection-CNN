import argparse
import csv
import torch
import networks
import utils
import kitti
import numpy as np
from skimage.transform import resize


EVAL_Y, EVAL_X = (375, 1220)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-path', required=True, help='Path to the KITTI dataset')
parser.add_argument('-saved_model', required=True, help='Path to the trained model')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
args = parser.parse_args()

# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Dataset
test_dataset = kitti.KITTI(args.path, subset='val', batch_size=1, augment=False, downsample=False)

# CNN
model = networks.MultiPurposeCNN().to(device)
model.load_state_dict(torch.load('{}'.format(args.saved_model)))
model.eval()


def evaluate():

    metrics = {'delta_1': [], 'delta_2': [], "delta_3": [], 'Rel': [], 'RMSE': [], "log10": []}
    for idx in range(len(test_dataset)):

        print("Predicting image {:05d} of {}".format(idx + 1, len(test_dataset)))

        # Load data
        sample = test_dataset[idx]
        images = sample['images']
        depth_targets = sample['depths']
        object_targets = sample['objects']

        # Forward pass
        predictions = model(torch.cat([images, torch.flip(images, dims=[3])]).to(device))['depths'].squeeze()
        depth_predictions = torch.mean(torch.stack([predictions[0], torch.flip(predictions[1], dims=[1])]), dim=0)

        # Convert to Numpy and up-sample
        depth_predictions = depth_predictions.detach().cpu().numpy()[:EVAL_Y, :EVAL_X]
        depth_targets = depth_targets.detach().cpu().numpy().squeeze()[:EVAL_Y, :EVAL_X]
        depth_predictions = resize(depth_predictions, depth_targets.shape, mode='constant', cval=1e-3,
                                   preserve_range=True, anti_aliasing=False).astype(np.float32)

        # Get metrics for depth task
        metrics_depth = utils.depth_metrics(depth_predictions, depth_targets)

        for key in metrics_depth.keys():
            metrics[key].append(metrics_depth[key])

        # Get metrics for specific objects: cars, people

    # Take the average
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])

    # Export to CSV
    with open('results/{}.csv'.format(args.saved_model.split('.')[0]), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == '__main__':
    evaluate()




