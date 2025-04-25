import torch
import torch.nn as nn


class SDSBDecoderLevel(nn.Module):
    def __init__(self, encoder_features, num_layers):
        super(SDSBDecoderLevel, self).__init__()
        self.upsample_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        total_channels = sum([feat.shape[1] for feat in encoder_features])
        self.fusion_conv1 = nn.Conv2d(total_channels, total_channels, kernel_size=1)
        self.scale_depth_conv = nn.Conv2d(total_channels, total_channels, kernel_size=1)
        self.scale_spatial_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=2, stride=2),
            nn.Conv2d(total_channels, total_channels, kernel_size=2, stride=2)
        )

    def forward(self, encoder_features):
        upsampled_features = [encoder_features[0]]
        for i in range(1, len(encoder_features)):
            upsampled_features.append(self.upsample_layers[i - 1](encoder_features[i]))
        fusion_vector = torch.cat(upsampled_features, dim=1)
        fusion_vector = self.fusion_conv1(fusion_vector)
        fusion_vector = self.scale_depth_conv(fusion_vector)
        output = self.scale_spatial_conv(fusion_vector)
        return output
    