# ------------------------------------------------------------------------------
# Dataset loaders
# ------------------------------------------------------------------------------

import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import torchvision
from torchvision import datasets

def prepare_cifar10_loaders(data_transform, train_batch_size=1, eval_batch_size=1):

  train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=data_transform)
  test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transform)

  train_sampler = RandomSampler(train_set)
  test_sampler = SequentialSampler(test_set)

  train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, sampler=train_sampler)
  test_loader = DataLoader(dataset=test_set, batch_size=eval_batch_size, sampler=test_sampler)

  return train_loader, test_loader

def prepare_cifar100_loaders(data_transform, train_batch_size=1, eval_batch_size=1):

  train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=data_transform)
  test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=data_transform)

  train_sampler = RandomSampler(train_set)
  test_sampler = SequentialSampler(test_set)

  train_loader = DataLoader(dataset=train_set, batch_size=train_batch_size, sampler=train_sampler)
  test_loader = DataLoader(dataset=test_set, batch_size=eval_batch_size, sampler=test_sampler)

  return train_loader, test_loader
