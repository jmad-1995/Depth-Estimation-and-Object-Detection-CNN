import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.transform import resize


class KITTI(Dataset):

    def __init__(self, path, subset='val', batch_size=1, augment=False, downsample=True):
        super().__init__()
        assert subset in ['train', 'val']

        # Create paths
        self.path = path
        self.depth_path = os.path.join(self.path, 'depth', 'processed_depth_maps', subset)
        self.image_path = os.path.join(self.path, 'depth', 'images', subset)

        # Get files
        self.depth_files = next(os.walk(self.depth_path))[2]
        np.random.shuffle(self.depth_files)
        self.image_files = self.depth_files

        # Set options
        self.batch_size = batch_size
        self.downsample = downsample

        # Determine pre-processing sequence
        if augment:
            self.transform = self._augment
        else:
            self.transform = self._preprocess

    def __getitem__(self, item):

        # File indices
        idx = item % self.__len__()
        _slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)

        images = []
        depths = []
        for image_file, depth_file in zip(self.image_files[_slice], self.depth_files[_slice]):

            # Import image and depth map
            image = np.array(Image.open(os.path.join(self.image_path, image_file)), np.float32) / 255.
            depth = np.array(Image.open(os.path.join(self.depth_path, depth_file)), np.float32) / 1000.
            if self.downsample:
                depth = resize(depth, (192, 640), mode='constant', cval=1e-3, preserve_range=True, anti_aliasing=False).astype(np.float32)

            # Pre-process or augment
            image, depth = self.transform(image, depth)

            # Append to batch
            images.append(image)
            depths.append(depth)

        return {'images': torch.stack(images), 'depths': torch.stack(depths), 'objects': []}

    def __len__(self):
        return len(self.depth_files) // self.batch_size

    def _augment(self, image, depth):

        # Horizontal Flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            depth = np.flip(depth, axis=1)

        # Channel swap
        if np.random.rand() > 0.5:
            image = image.transpose((2, 0, 1))
            image = image[np.random.permutation(3)]
            image = image.transpose((1, 2, 0))

        # Color jitter
        image = Image.fromarray(np.uint8(image * 255))
        image = transforms.Compose([transforms.ColorJitter(brightness=0.25, contrast=0.25)])(image)
        image = np.array(image, np.float32) / 255.

        # Pad
        image, depth = self._preprocess(image, depth)

        return image, depth

    def _preprocess(self, image, depth):

        # Pad image
        pad_x = 1280 - image.shape[1]
        pad_y = 384 - image.shape[0]
        image = np.pad(image, pad_width=((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=1e-3)
        if not self.downsample:
            depth = np.pad(depth, pad_width=((0, pad_y), (0, pad_x)), mode='constant', constant_values=1e-3)

        # Create tensor
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        depth = torch.from_numpy(depth.copy()).view(1, *depth.shape)

        return image, depth


if __name__ == '__main__':

    p = '/media/awmagsam/HDD/Datasets/KITTI'
    dataset = KITTI(p, augment=True, batch_size=5)