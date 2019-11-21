import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from visdom import Visdom


class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')

    def show_image(self, model, test_dataset, train_dataset, device):

        model.eval()

        #####################################################################################################
        # Validation
        idx = np.random.randint(len(test_dataset))
        sample = test_dataset[idx]

        # Inference
        rgb = sample['images'].to(device)
        depth = sample['depths'].to(device)
        prediction = model(rgb)

        # Choose images to display and transfer to CPU
        batch_idx = np.random.randint(rgb.shape[0])
        image = rgb[batch_idx].cpu()
        target = depth[batch_idx].cpu()
        prediction = prediction[batch_idx].cpu()

        # Get normalized 2D depth maps HxW
        target = np.clip(target.detach().numpy().transpose((1, 2, 0)).squeeze() / 10., 0, 1)
        prediction = np.clip(prediction.detach().numpy().transpose((1, 2, 0)).squeeze() / 10., 0, 1)

        # Colorize images
        cmapper = cm.get_cmap('plasma')
        target = cmapper(target).transpose((2, 0, 1))
        prediction = cmapper(prediction).transpose((2, 0, 1))

        if 'rgb' not in self.plots:
            self.plots['rgb'] = self.viz.image(image, env=self.env, opts={'store_history': True, 'caption': 'RGB'})
        else:
            self.viz.image(image, env=self.env, win=self.plots['rgb'],
                           opts={'store_history': True, 'caption': 'RGB'})

        if 'target' not in self.plots:
            self.plots['target'] = self.viz.image(torch.from_numpy(target.copy()), env=self.env,
                                                  opts={'store_history': True, 'caption': 'Target'})
        else:
            self.viz.image(torch.from_numpy(target), env=self.env, win=self.plots['target'],
                           opts={'store_history': True, 'caption': 'Target'})

        if 'prediction' not in self.plots:
            self.plots['prediction'] = self.viz.image(torch.from_numpy(prediction.copy()), env=self.env,
                                                      opts={'store_history': True, 'caption': 'Prediction'})
        else:
            self.viz.image(torch.from_numpy(prediction), env=self.env, win=self.plots['prediction'],
                           opts={'store_history': True, 'caption': 'Prediction'})

        #####################################################################################################
        # Training
        idx = np.random.randint(len(train_dataset))
        sample = train_dataset[idx]

        # Inference
        rgb = sample['images'].to(device)
        depth = sample['depths'].to(device)
        prediction = model(rgb)

        # Choose images to display and transfer to CPU
        batch_idx = np.random.randint(rgb.shape[0])
        image = rgb[batch_idx].cpu()
        target = depth[batch_idx].cpu()
        prediction = prediction[batch_idx].cpu()

        # Get normalized 2D depth maps HxW
        target = np.clip(target.detach().numpy().transpose((1, 2, 0)).squeeze() / 10., 0, 1)
        prediction = np.clip(prediction.detach().numpy().transpose((1, 2, 0)).squeeze() / 10., 0, 1)

        # Colorize images
        cmapper = cm.get_cmap('plasma')
        target = cmapper(target).transpose((2, 0, 1))
        prediction = cmapper(prediction).transpose((2, 0, 1))

        if 'rgb_t' not in self.plots:
            self.plots['rgb_t'] = self.viz.image(image, env=self.env, opts={'store_history': True, 'caption': 'RGB_t'})
        else:
            self.viz.image(image, env=self.env, win=self.plots['rgb_t'],
                           opts={'store_history': True, 'caption': 'RGB_t'})

        if 'target_t' not in self.plots:
            self.plots['target_t'] = self.viz.image(torch.from_numpy(target.copy()), env=self.env,
                                                  opts={'store_history': True, 'caption': 'Target_t'})
        else:
            self.viz.image(torch.from_numpy(target), env=self.env, win=self.plots['target_t'],
                           opts={'store_history': True, 'caption': 'Target_t'})

        if 'prediction_t' not in self.plots:
            self.plots['prediction_t'] = self.viz.image(torch.from_numpy(prediction.copy()), env=self.env,
                                                      opts={'store_history': True, 'caption': 'Prediction_t'})
        else:
            self.viz.image(torch.from_numpy(prediction), env=self.env, win=self.plots['prediction_t'],
                           opts={'store_history': True, 'caption': 'Prediction_t'})


def visualize_detections(image, bboxes):
    pass


def visualize_depth(image, depth):
    pass