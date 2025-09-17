import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def create_data(resize = (64,64)):

    transform = transforms.Compose([
        transforms.Resize((resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])]) ## Noralizaos a -1 a 1

    train_dataset = datasets.CIFAR10(root="./data", train=True,download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False,download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    return train_loader , test_loader

