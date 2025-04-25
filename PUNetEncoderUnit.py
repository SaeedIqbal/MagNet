import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel

class PUNetEncoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, d, r):
        super(PUNetEncoderUnit, self).__init__()
        self.sdc = SDC(in_channels, out_channels, dilation_rates, d, r)
        self.smoothed_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.sdc(x)
        x = self.smoothed_conv(x)
        return x