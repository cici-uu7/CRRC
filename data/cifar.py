# data/cifar.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Cifar:
    def __init__(self, batch_size, num_workers):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        self.test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

        self.train = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.test = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
