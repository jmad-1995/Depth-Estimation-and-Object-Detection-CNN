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

    metrics = {'_': 'depth maps', 'delta_1': [], 'delta_2': [], "delta_3": [], 'Rel': [], 'RMSE': [], "log10": []}
    people = {'_': 'people', 'delta_1': [], 'delta_2': [], "delta_3": [], 'Rel': [], 'RMSE': [], "log10": []}
    vehicles = {'_': 'vehicles', 'delta_1': [], 'delta_2': [], "delta_3": [], 'Rel': [], 'RMSE': [], "log10": []}
    for idx in range(len(test_dataset)):

        print("Predicting image {:05d} of {}".format(idx + 1, len(test_dataset)))

        # Load data
        sample = test_dataset[idx]
        images = sample['images']
        depth_targets = sample['depths']
        object_targets = sample['objects']

        # Forward pass
        depth_predictions = model(images.to(device))['depths'].squeeze()

        # Convert to Numpy and up-sample
        depth_predictions = depth_predictions.detach().cpu().numpy()
        depth_targets = depth_targets.detach().cpu().numpy().squeeze()[:EVAL_Y, :EVAL_X]
        depth_predictions = resize(depth_predictions, depth_targets.shape, mode='constant', cval=1e-3,
                                   preserve_range=True, anti_aliasing=False).astype(np.float32)[:EVAL_Y, :EVAL_X]

        # Get metrics for depth task
        metrics_depth = utils.depth_metrics(depth_predictions, depth_targets)
        for key in metrics_depth.keys():
            metrics[key].append(metrics_depth[key])

        # Calculate object specific metrics
        if object_targets:

            bboxes = object_targets['bboxes']
            classes = object_targets['classes']

            # Generate targets and predictions for detection specific depths
            object_depths_targets = utils.get_object_depths(bboxes, classes, depth_targets)
            object_depths_predictions = utils.get_object_depths(bboxes, classes, depth_predictions)

            # Get metrics for people class
            if object_depths_targets['1']:
                metrics_depth = utils.depth_metrics(object_depths_predictions['1'], object_depths_targets['1'])
                for key in metrics_depth.keys():
                    people[key].append(metrics_depth[key])

            # Get metrics for vehicle class
            if object_depths_targets['2']:
                metrics_depth = utils.depth_metrics(object_depths_predictions['2'], object_depths_targets['2'])
                for key in metrics_depth.keys():
                    vehicles[key].append(metrics_depth[key])

    # Take the average
    for key in metrics.keys():
        if key != '_':
            metrics[key] = np.mean(metrics[key])
            people[key] = np.mean(people[key])
            vehicles[key] = np.mean(vehicles[key])

    # Export to CSV
    with open('results/results.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
        writer.writerow(people)
        writer.writerow(vehicles)


if __name__ == '__main__':
    evaluate()



