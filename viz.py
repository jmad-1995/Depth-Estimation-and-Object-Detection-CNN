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

    def show_images(self, model, train_dataset, valid_dataset, device):

        model.eval()
        post = lambda x: np.clip(x.detach().numpy().transpose((1, 2, 0)).squeeze() / 20, 0., 1.)

        #####################################################################################################
        # Validation
        idx = np.random.randint(len(valid_dataset))
        sample = valid_dataset[idx]

        # Inference
        images = sample['images'].to(device)
        depths = sample['depths'].to(device)
        prediction = model(images)
        prediction = prediction['depths']

        # Choose images to display and transfer to CPU
        batch_idx = np.random.randint(images.shape[0])
        image = images[batch_idx].cpu()
        target = depths[batch_idx].cpu()
        prediction = prediction[batch_idx].cpu()

        # Get normalized 2D depth maps HxW
        target = post(target)
        prediction = post(prediction)

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
        images = sample['images'].to(device)
        depths = sample['depths'].to(device)
        prediction = model(images)
        prediction = prediction['depths']

        # Choose images to display and transfer to CPU
        batch_idx = np.random.randint(images.shape[0])
        image = images[batch_idx].cpu()
        target = depths[batch_idx].cpu()
        prediction = prediction[batch_idx].cpu()

        # Get normalized 2D depth maps HxW
        target = post(target)
        prediction = post(prediction)

        # Colorize images
        cmapper = cm.get_cmap('plasma')
        target = cmapper(target).transpose((2, 0, 1))
        prediction = cmapper(prediction).transpose((2, 0, 1))

        if 'rgb_t' not in self.plots:
            self.plots['rgb_t'] = self.viz.image(image, env=self.env, opts={'store_history': True, 'caption': 'RGB_train'})
        else:
            self.viz.image(image, env=self.env, win=self.plots['rgb_t'],
                           opts={'store_history': True, 'caption': 'RGB_train'})

        if 'target_t' not in self.plots:
            self.plots['target_t'] = self.viz.image(torch.from_numpy(target.copy()), env=self.env,
                                                    opts={'store_history': True, 'caption': 'Target_train'})
        else:
            self.viz.image(torch.from_numpy(target), env=self.env, win=self.plots['target_t'],
                           opts={'store_history': True, 'caption': 'Target_train'})

        if 'prediction_t' not in self.plots:
            self.plots['prediction_t'] = self.viz.image(torch.from_numpy(prediction.copy()), env=self.env,
                                                      opts={'store_history': True, 'caption': 'Prediction_t'})
        else:
            self.viz.image(torch.from_numpy(prediction), env=self.env, win=self.plots['prediction_t'],
                           opts={'store_history': True, 'caption': 'Prediction_train'})