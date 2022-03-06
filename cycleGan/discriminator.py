import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, instance_norm=True):
        super().__init__()
        if instance_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True,
                          padding_mode='reflect'),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True,
                          padding_mode='reflect'),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, channels_list=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        for idx, channel in enumerate(channels_list):
            if idx == 0:
                instance_norm = False
            else:
                instance_norm = True
            if idx == len(channels_list) - 1:
                stride = 1
            else:
                stride = 2
            layers.append(ConvBlock(in_channels, channel, stride=stride, instance_norm=instance_norm))
            in_channels = channel
        layers.append(nn.Conv2d(channels_list[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
