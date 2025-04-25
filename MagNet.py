import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel

class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Example configuration for SDC blocks in encoder
        in_channels = 3
        out_channels = 64
        dilation_rates = [1, 2, 3]
        d = 2
        r = 3
        for _ in range(4):
            self.encoder.append(PUNetEncoderUnit(in_channels, out_channels, dilation_rates, d, r))
            in_channels = out_channels
            out_channels *= 2

        in_channels = out_channels // 2
        out_channels = in_channels // 2
        for _ in range(4):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )
            )
            in_channels = out_channels
            out_channels = in_channels // 2

        self.sdsb = None
        self.prb = None

    def forward(self, x):
        encoder_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_features.append(x)

        num_layers = len(encoder_features)
        self.sdsb = SDSBDecoderLevel(encoder_features, num_layers)
        sdsb_output = self.sdsb(encoder_features)

        decoder_features = []
        for i, decoder_layer in enumerate(self.decoder):
            if i == 0:
                x = decoder_layer(sdsb_output)
            else:
                x = decoder_layer(x + sdsb_output)
            decoder_features.append(x)

        self.prb = PRB(decoder_features)
        output = self.prb(decoder_features)
        return output


