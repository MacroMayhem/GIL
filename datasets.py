__author__ = "Aditya Singh"
__version__ = "0.1"

import torchvision
import os
import pickle
import sys
import numpy as np
from PIL import Image


class SubLoader(torchvision.datasets.CIFAR100):
    def __init__(self, *args, include_list=[], **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()


class BufferDataset(torchvision.datasets.VisionDataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, transform=None, target_transform=None, buffer_size=50):
        super(BufferDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        downloaded_list = self.train_list
        self.buffer_size = buffer_size
        self.data = []
        self.targets = []
        self.all_data = []
        self.all_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.all_data.append(entry['data'])
                if 'labels' in entry:
                    self.all_targets.extend(entry['labels'])
                else:
                    self.all_targets.extend(entry['fine_labels'])

        self.all_data = np.vstack(self.all_data).reshape(-1, 3, 32, 32)
        self.all_data = self.all_data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def append_data(self, batch_classes):
        labels = np.array(self.all_targets)
        for cls in batch_classes:
            indices = np.where(labels == cls)[0]
            np.random.shuffle(indices)
            self.data.extend(self.all_data[indices[0:self.buffer_size]])
            self.targets.extend([cls for i in range(self.buffer_size)])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
