from torch.utils.data import Dataset


class KITTIDataset(Dataset):

    def __init__(self, path, batch_size, augment=False):
        super().__init__()

        self.path = path
        self.files = []
        self.batch_size = batch_size

        if augment:
            self.transform = []
        else:
            self.transform = []

    def __getitem__(self, item):
        return {'images': [], 'depths': [], 'objects': []}

    def __len__(self):
        return len(self.files) // self.batch_size

    def _augment(self, image, depth, objects):
        return image, depth, objects