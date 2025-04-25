import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from sdsb_decoder import SDSBDecoderLevel

class PRB(nn.Module):
    def __init__(self, decoder_features):
        super(PRB, self).__init__()
        self.upsample_layers = nn.ModuleList()
        for i in range(len(decoder_features) - 1):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.fusion_conv1 = nn.Conv2d(sum([feat.shape[1] for feat in decoder_features]),
                                      sum([feat.shape[1] for feat in decoder_features]), kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(sum([feat.shape[1] for feat in decoder_features]), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, decoder_features):
        upsampled_features = [decoder_features[0]]
        for i in range(1, len(decoder_features)):
            upsampled_features.append(self.upsample_layers[i - 1](decoder_features[i]))
        fusion_vector = torch.cat(upsampled_features, dim=1)
        fusion_vector = self.fusion_conv1(fusion_vector)
        output = self.final_conv(fusion_vector)
        return output
