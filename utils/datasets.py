# ------------------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------------------

import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import torchvision
from torchvision import datasets
from torchvision.io import read_image

def prepare_cifar10_loaders(data_transform, train_batch_size=1, eval_batch_size=1):
    train_set = datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=data_transform
        )
    test_set = datasets.CIFAR10(
        root="./data", 
        train=False, 
        download=True, 
        transform=data_transform
        )

    train_sampler = RandomSampler(train_set)
    test_sampler = SequentialSampler(test_set)

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=train_batch_size, 
        sampler=train_sampler
        )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=eval_batch_size, 
        sampler=test_sampler
        )

    return train_loader, test_loader

def prepare_cifar100_loaders(data_transform, train_batch_size=1, eval_batch_size=1):
    train_set = datasets.CIFAR100(
        root="./data", 
        train=True, 
        download=True, 
        transform=data_transform
        )
    test_set = datasets.CIFAR100(
        root="./data", 
        train=False, 
        download=True, 
        transform=data_transform
        )

    train_sampler = RandomSampler(train_set)
    test_sampler = SequentialSampler(test_set)

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=train_batch_size, 
        sampler=train_sampler
        )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=eval_batch_size, 
        sampler=test_sampler
        )

    return train_loader, test_loader

class ImagenetCalibrationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = []

        # get image file names
        for file_name in os.listdir(path=img_dir):
            if file_name [-5:] == '.JPEG':
                self.img_names.append(file_name)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, idx

def prepare_imagenet_loaders(data_transform, train_batch_size=1, eval_batch_size=1):
    train_set = ImagenetCalibrationDataset("./data/train", transform=data_transform)
    train_sampler = SequentialSampler(train_set)
    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=eval_batch_size, 
        sampler=train_sampler
        )

    test_set = datasets.ImageNet(root="./data", split='val', transform=data_transform)
    test_sampler = SequentialSampler(test_set)
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=eval_batch_size, 
        sampler=test_sampler
        )

    return train_loader, test_loader
