from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms


class PairDataset(Dataset):
    """"""
    def __init__(self,
                 dataset='cifar10',
                 root='data',
                 transform=None,
                 download=True):
        """
        Paired dataset, generate mini-batche pairs on CIFAR10 training set.
        :param dataset: name of dataset, str.
        :param root: directory of dataset.
        :param transform: transform applied on input x.
        :param download: if download dataset.
        """
        assert dataset in ['cifar10', 'cifar100']
        if dataset == 'cifar10':
            self.dataset = CIFAR10(root=root, transform=None, download=download)  # no transform here
        elif dataset == 'cifar100':
            self.dataset = CIFAR100(root=root, transform=None, download=download)
        else:
            raise Exception('dataset {} not available, should be one of [cifar10, cifar100]'.format(dataset))

        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset.data[idx], self.dataset.targets[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair

    def __len__(self):
        return len(self.dataset)


class AugPairDataset(Dataset):
    def __init__(self,
                 dataset='cifar10',
                 root='data',
                 transform=None,
                 download=True,
                 aug_list=[],
                 width=1,
                 depth=1,
                 aug_severity=1):
        """
        Augmented paired dataset.
        :param dataset: name of dataset, str.
        :param root: directory of dataset.
        :param transform: pre transform of input x (not including ToTensor())
        :param download: if download dataset.
        :param aug_list: list of augmentations available.
        :param width: width of augmentation mixture.
        :param depth: depth of augmentation mixture.
        :param aug_severity: severity level of augmentation.
        """
        assert dataset in ['cifar10', 'cifar100']
        if dataset == 'cifar10':
            self.dataset = CIFAR10(root=root, transform=None, download=download)
        elif dataset == 'cifar100':
            self.dataset = CIFAR100(root=root, transform=None, download=download)
        else:
            raise Exception('dataset {} not available, should be one of [cifar10, cifar100]'.format(dataset))

        self.transform = transform
        self.aug_list = aug_list
        self.width = width
        self.depth = depth
        self.aug_severity = aug_severity
        self.to_tensor = transforms.ToTensor()

    def _aug(self, x):
        ws = np.float32(np.random.dirichlet([1] * self.width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(self.to_tensor(x))
        for i in range(self.width):  # size of composed augmentations set
            image_aug = x.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):  # compose one augmentation with depth number of single aug operation.
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.to_tensor(image_aug)

        mixed = (1 - m) * self.to_tensor(x) + m * mix
        return mixed

    def __getitem__(self, i):
        img, target = self.dataset.data[i], self.dataset.targets[i]
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)  # PIL image
        x_clean = self.to_tensor(img)
        x_aug = self.to_tensor(self._aug(img))
        return torch.stack([x_clean, x_aug]), target

    def __len__(self):
        return len(self.dataset)
