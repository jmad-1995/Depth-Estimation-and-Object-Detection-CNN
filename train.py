import argparse
import torch
import torch.optim as optim
import numpy as np
import networks
import utils
import kitti
import viz


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-path', required=True, help='Path to the KITTI dataset')
parser.add_argument('-batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('-n_iters', default=1e4, type=int, help='Number of training iterations')
parser.add_argument('-test_freq', default=1000, type=int, help='Frequency of testing and plotting validation results')
parser.add_argument('-save_freq', default=1000, type=int, help='Frequency to save trained model')
parser.add_argument('-prefix', default='model', help='Save name prefix for trained models')
parser.add_argument('-lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
args = parser.parse_args()

# CUDA device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Dataset
train_dataset = kitti.KITTI(args.path, subset='train', batch_size=args.batch_size, augment=True)
test_dataset = kitti.KITTI(args.path, subset='val', batch_size=args.batch_size)

# CNN
model = networks.MultiPurposeCNN().to(device)
# model.load_state_dict(torch.load('models/base.pt'))
model.train()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

# Loss functions
criterion_depth = utils.depth_loss
criterion_objects = utils.objects_loss

# Visdom writer
writer = viz.VisdomPlotter()


def train():
    model.train()

    running_loss = []
    for idx in range(args.n_iters):
        model.train()
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

        # Loss calculation
        loss_depth = criterion_depth(depth_predictions, depth_targets.to(device))
        loss = loss_depth
        running_loss.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step(idx)
        lr = scheduler.get_lr()[-1]

        print("Iteration : {:06d}\t\tTrain Loss : {:.3f}\t\t Learning Rate: {:.5f}".format(idx + 1, np.mean(running_loss), lr), end='\r')

        # Test
        if (idx + 1) % args.test_freq == 0:
            test_loss = test()

            # Visualize results on the server
            writer.plot('loss', 'train', 'Depth loss', idx, np.mean(running_loss))
            writer.plot('loss', 'validation', 'Depth loss', idx, test_loss)
            writer.show_images(model, train_dataset, test_dataset, device)

            print("\nIteration : {:06d}\t\tValidation Loss : {:.3f}".format(idx + 1, test_loss), end='\n\n')

        # Save model
        if (idx + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), "models/{}-{}.pt".format(args.prefix, idx + 1))


def test():
    model.eval()

    running_loss = []
    for idx in np.random.choice(np.random.permutation(len(test_dataset)), args.test_freq):

        # Load data
        sample = test_dataset[idx]
        images = sample['images']
        depth_targets = sample['depths']
        object_targets = sample['objects']

        # Forward pass
        predictions = model(images.to(device))
        depth_predictions = predictions['depths']

        # Loss calculation
        loss_depth = criterion_depth(depth_predictions, depth_targets.to(device))
        loss = loss_depth
        running_loss.append(loss.item())

    return np.mean(running_loss)


if __name__ == '__main__':
    train()
