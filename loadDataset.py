import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel


# 1. Load the BreakHis dataset
class BreakHisDataset:
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    def get_loader(self, batch_size=32, shuffle=True):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)