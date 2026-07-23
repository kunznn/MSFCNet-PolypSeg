from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PureUNetDoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PureUNetUpBlock2d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = PureUNetDoubleConv2d(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class PureUNet2D(nn.Module):
    """Standard 6-level 2D U-Net with Conv-BN-ReLU blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256, 512, 512),
    ) -> None:
        super().__init__()
        if len(features) != 6:
            raise ValueError("PureUNet2D expects 6 feature levels, e.g. (32, 64, 128, 256, 512, 512).")

        self.enc1 = PureUNetDoubleConv2d(in_channels, features[0])
        self.enc2 = PureUNetDoubleConv2d(features[0], features[1])
        self.enc3 = PureUNetDoubleConv2d(features[1], features[2])
        self.enc4 = PureUNetDoubleConv2d(features[2], features[3])
        self.enc5 = PureUNetDoubleConv2d(features[3], features[4])
        self.bottleneck = PureUNetDoubleConv2d(features[4], features[5])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec5 = PureUNetUpBlock2d(features[5], features[4], features[4])
        self.dec4 = PureUNetUpBlock2d(features[4], features[3], features[3])
        self.dec3 = PureUNetUpBlock2d(features[3], features[2], features[2])
        self.dec2 = PureUNetUpBlock2d(features[2], features[1], features[1])
        self.dec1 = PureUNetUpBlock2d(features[1], features[0], features[0])
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bottleneck(self.pool(e5))

        d5 = self.dec5(b, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)
