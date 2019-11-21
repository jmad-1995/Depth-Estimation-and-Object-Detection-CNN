import os
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset


class KITTI(Dataset):

    def __init__(self, path, subset='val', batch_size=1, augment=False):
        super().__init__()
        assert subset in ['train', 'val']

        self.path = path
        self.depth_path = os.path.join(self.path, subset, 'depth')
        self.image_path = os.path.join(self.path, subset, 'images')

        self.depth_files = next(os.walk(self.depth_path))[2]
        self.image_files = next(os.walk(self.image_files))[2]

        self.batch_size = batch_size

        if augment:
            self.transform = self._augment
        else:
            self.transform = self._preprocess

    def __getitem__(self, item):

        # File indices
        idx = item % self.__len__()
        _slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)

        for image_file, depth_file in zip(self.image_files[_slice], self.depth_files[_slice]):

            # Import image and depth map
            image = np.array(Image.open(os.path.join(self.image_path, image_file)), np.float32)
            depth = np.array(Image.open(os.path.join(self.depth_path, depth_file)), np.float32) / 1000.

        return {'images': [], 'depths': [], 'objects': []}

    def __len__(self):
        return len(self.depth_files) // self.batch_size

    @staticmethod
    def _augment(image, depth):

        # Horizontal Flip

        # Channel swap

        # Color jitter

        # Pad

        return image, depth

    @staticmethod
    def _preprocess(image, depth):

        # Pad

        return image, depth


if __name__ == '__main__':

    p = '/media/awmagsam/HDD/Datasets/KITTI/object'

    dataset = KITTI(p)

    sample = dataset[1]

