import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel

class SDC(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, d, r):
        super(SDC, self).__init__()
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels * d, kernel_size=1)
        self.depthwise_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.depthwise_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels * d, in_channels * d, kernel_size=3, dilation=rate, padding=rate,
                              groups=in_channels * d),
                    nn.Conv2d(in_channels * d, in_channels * d, kernel_size=(3, 1), padding=(1, 0),
                              groups=in_channels * d),
                    nn.Conv2d(in_channels * d, in_channels * d, kernel_size=(1, 3), padding=(0, 1),
                              groups=in_channels * d)
                )
            )
        self.pointwise_conv2 = nn.Conv2d(in_channels * d * len(dilation_rates), out_channels, kernel_size=1)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        outputs = []
        for depthwise_conv in self.depthwise_convs:
            outputs.append(depthwise_conv(x))
        x = torch.cat(outputs, dim=1)
        x = self.pointwise_conv2(x)
        return x