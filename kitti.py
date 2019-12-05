import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.transform import resize


LABEL_MAP = {'Pedestrian': 1, 'Car': 2, 'Truck': 2, 'Van': 2}


class KITTI(Dataset):

    def __init__(self, path, subset='val', batch_size=1, augment=False, downsample=True):
        super().__init__()
        assert subset in ['train', 'val']

        # Create paths
        self.path = path
        self.depth_path = os.path.join(self.path, 'depth', 'processed_depth_maps', subset)
        self.image_path = os.path.join(self.path, 'depth', 'images', subset)
        self.object_path = os.path.join(self.path, 'object')

        # Get files
        self.depth_files = sorted(next(os.walk(self.depth_path))[2])
        # np.random.shuffle(self.depth_files)
        self.image_files = self.depth_files

        # Get object files
        self.object_labels = sorted(next(os.walk(os.path.join(self.object_path, 'labels')))[2])
        self.object_images = sorted(next(os.walk(os.path.join(self.object_path, 'images')))[2])
        count = 0
        for file in self.depth_files:
            if file in self.object_images:
                count += 1
        print("There are {} overlapping files with depth + object labels".format(count))

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
        objects = {}
        for image_file, depth_file in zip(self.image_files[_slice], self.depth_files[_slice]):

            # Import image and depth map
            image = np.array(Image.open(os.path.join(self.image_path, image_file)), np.float32) / 255.
            depth = np.array(Image.open(os.path.join(self.depth_path, depth_file)), np.float32) / 1000.
            if self.downsample:
                depth = resize(depth, (192, 640), mode='constant', cval=1e-3, preserve_range=True, anti_aliasing=False).astype(np.float32)

            if depth_file in self.object_images:
                bboxes, classes = self._read_object_labels(os.path.join(self.object_path, 'labels',
                                                                        depth_file[:-4] + '.txt'))
                objects['bboxes'] = bboxes
                objects['classes'] = classes

            # Pre-process or augment
            image, depth = self.transform(image, depth)

            # Append to batch
            images.append(image)
            depths.append(depth)

        return {'images': torch.stack(images), 'depths': torch.stack(depths), 'objects': objects}

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

    @staticmethod
    def _read_object_labels(file):

        with open(file, 'r') as f:
            string = f.readlines()

        bboxes = []
        classes = []
        for item in string:
            split = item.split(' ')
            if split[0] in LABEL_MAP.keys():
                _class = LABEL_MAP[split[0]]
                x1 = float(split[4])
                y1 = float(split[5])
                x2 = float(split[6])
                y2 = float(split[7])

                bboxes.append([y1, x1, y2, x2])
                classes.append(_class)
            else:
                continue

        return np.array(bboxes).astype(int), np.array(classes)


if __name__ == '__main__':

    p = r'A:\Deep Learning Datasets\KITTI'
    dataset = KITTI(p, subset='val', augment=False, batch_size=1)

    from matplotlib.patches import Rectangle

    peop = 0
    veh = 0
    for idx in range(len(dataset)):
        print("Sample {:05d}".format(idx + 1))
        s = dataset[idx]
        i = s['images']
        o = s['objects']

        if o:
            # fig, axes = plt.subplots()
            # plt.imshow(i.cpu().squeeze().permute(1, 2, 0).numpy())
            #
            # for b in o['bboxes']:
            #         r = Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], color=(1, 0, 0), fill=None)
            #         axes.add_patch(r)
            #
            # plt.show()
            veh += np.count_nonzero(o['classes'] == 2)
            peop += np.count_nonzero(o['classes'] == 1)

    print("Vehicles = ", veh)
    print("People = ", peop)