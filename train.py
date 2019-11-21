import argparse
import torch
import torch.optim as optim
import numpy as np
import networks
import utils
import kitti


# Argument parser
parser = argparse.ArgumentParser()
args = parser.parse_args()

# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Dataset
train_dataset = kitti.KITTI(args.path, batch_size=args.batch_size, augment=True)
test_dataset = kitti.KITTI(args.path, batch_size=args.batch_size)

# CNN
model = networks.MultiPurposeCNN().to(device)
model.train()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

# Loss functions
criterion_depth = utils.depth_loss
criterion_objects = utils.objects_loss


def train():
    model.train()

    running_loss = []
    for idx in range(args.n_iters):

        # Zero gradient
        optimizer.zero_grad()

        # Load data
        sample = train_dataset[idx]
        images = sample['images']
        depth_targets = sample['depths']
        object_targets = sample['objects']

        # Forward pass
        predictions = model(images.to(device))
        depth_predictions = predictions['depths']
        object_predictions = predictions['objects']

        # Loss calculation
        loss_depth = criterion_depth(depth_predictions, depth_targets)
        loss_objects = criterion_objects(object_predictions, object_targets)
        loss = loss_depth + loss_objects
        running_loss.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Test
        if (idx + 1) % args.test_freq == 0:
            test_loss = test()

            # Visualize results on the server

        # Save model
        if (idx + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), "models/{}-{}.pt".format(args.prefix, idx))

        print("Iteration : {:06d}\t\tTrain Loss : {:.3f}".format(idx+1, np.mean(running_loss)), end='\r')


def test():
    model.eval()

    running_loss = []
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

        # Loss calculation
        loss_depth = criterion_depth(depth_predictions, depth_targets)
        loss_objects = criterion_objects(object_predictions, object_targets)
        loss = loss_depth + loss_objects
        running_loss.append(loss.item())

    return np.mean(running_loss)


if __name__ == '__main__':
    train()
