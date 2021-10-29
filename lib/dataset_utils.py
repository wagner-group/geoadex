'''
Dataset and DataLoader adapted from
https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
'''

import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def load_mnist(batch_size,
               data_dir='./data',
               val_size=0.1,
               shuffle=True,
               seed=1):
    """Load MNIST data into train/val/test data loader"""

    num_workers = 4

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist_all(
        data_dir=data_dir, val_size=val_size, shuffle=shuffle, seed=seed)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def load_mnist_all(data_dir='./data', val_size=0.1, shuffle=True, seed=1):
    """Load entire MNIST dataset into tensor"""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    x, y = next(iter(trainloader))
    x_test, y_test = next(iter(testloader))

    if val_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x.numpy(), y.numpy(), test_size=val_size, shuffle=shuffle,
            random_state=seed, stratify=y)
        return ((torch.tensor(x_train), torch.tensor(y_train)),
                (torch.tensor(x_valid), torch.tensor(y_valid)), (x_test, y_test))
    return ((x, y), (x_test, y_test))
