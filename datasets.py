__author__ = "Aditya Singh"
__version__ = "0.1"

import torchvision
import torch
import os
import pickle
import sys
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(dataset='mnist', accepted_class_labels=[], batch_size=64, random_seed=42
                           , valid_size=0.1, shuffle=True, num_workers=4, pin_memory=True):
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset.lower() == 'mnist':
        train_dataset = MNISTSubLoader(root='./data', train=True, download=True, transform=train_transform, include_list=accepted_class_labels)
        valid_dataset = MNISTSubLoader(root='./data', train=True, download=True, transform=valid_transform, include_list=accepted_class_labels)
    elif dataset.lower() == 'cifar100':
        train_dataset = CIFAR100SubLoader(root='./data', train=True, download=True, transform=train_transform, include_list=accepted_class_labels)
        valid_dataset = CIFAR100SubLoader(root='./data', train=True, download=True, transform=valid_transform, include_list=accepted_class_labels)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def get_test_loader(dataset='mnist', accepted_class_labels=[], batch_size=64, num_workers=4, pin_memory=True):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if dataset.lower() == 'mnist':
        train_dataset = MNISTSubLoader(root='./data', train=False, download=True, transform=test_transform
                                       , include_list=accepted_class_labels)
    elif dataset.lower() == 'cifar100':
        train_dataset = CIFAR100SubLoader(root='./data', train=False, download=True, transform=test_transform
                                          , include_list=accepted_class_labels)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size
                                              , num_workers=num_workers, pin_memory=pin_memory)
    return test_loader


class MNISTSubLoader(torchvision.datasets.MNIST):
    def __init__(self, *args, include_list=[], **kwargs):
        super(MNISTSubLoader, self).__init__(*args, **kwargs)

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()


class CIFAR100SubLoader(torchvision.datasets.CIFAR100):
    def __init__(self, *args, include_list=[], **kwargs):
        super(CIFAR100SubLoader, self).__init__(*args, **kwargs)

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
