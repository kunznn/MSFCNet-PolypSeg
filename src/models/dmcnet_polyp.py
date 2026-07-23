"""Adapted from pancreas-segmentation DMCNet2d (DMSC + DMRC modules),
modified for 2D RGB binary polyp segmentation (in_channels=3, out_channels=1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelGate2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ConvBNReLU2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConvBNReLU2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBN2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DMRC2d(nn.Module):
    """Dynamic multi-resolution convolution block from DMC-Net."""

    def __init__(self, channels: int, pool_size: int = 4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.conv_full = ConvBNReLU2d(channels, channels, kernel_size=3)
        self.conv_low = ConvBN2d(channels, channels, kernel_size=3)
        self.conv_pixel = ConvBN2d(channels, channels, kernel_size=1)
        self.conv_spatial = ConvBNReLU2d(channels, channels, kernel_size=3)
        self.channel_gate = ChannelGate2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = self.conv_full(x)
        low = self.pool(x)
        low = self.conv_low(low)
        low = F.interpolate(low, size=x.shape[-2:], mode="nearest")
        pixel = self.conv_pixel(x)
        spatial_attention = torch.sigmoid(low + pixel)
        calibrated = full * spatial_attention
        spatial_features = self.conv_spatial(calibrated)
        return self.channel_gate(spatial_features)


class DMSC2d(nn.Module):
    """Dynamic multi-scale convolution block from DMC-Net."""

    def __init__(self, channels: int, conv5_mode: str = "depthwise"):
        super().__init__()
        if conv5_mode not in {"depthwise", "standard"}:
            raise ValueError("conv5_mode must be 'depthwise' or 'standard'.")
        conv5 = DepthwiseSeparableConvBNReLU2d if conv5_mode == "depthwise" else ConvBNReLU2d
        self.conv3_1 = ConvBNReLU2d(channels, channels, kernel_size=3)
        self.conv3_2 = ConvBNReLU2d(channels, channels, kernel_size=3)
        self.conv5_1 = conv5(channels, channels, kernel_size=5)
        self.conv5_2 = conv5(channels, channels, kernel_size=5)
        self.channel_gate = ChannelGate2d(channels * 2)
        self.reduce = ConvBNReLU2d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small = self.conv3_2(self.conv3_1(x))
        large = self.conv5_2(self.conv5_1(x))
        multi_scale = torch.cat([small, large], dim=1)
        calibrated = self.channel_gate(multi_scale)
        return self.reduce(calibrated)


class DMCBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv5_mode: str = "depthwise"):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dmsc = DMSC2d(out_channels, conv5_mode=conv5_mode)
        self.dmrc = DMRC2d(out_channels, pool_size=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.dmsc(x)
        return self.dmrc(x)


class UpBlock2d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, conv5_mode: str = "depthwise"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block = DMCBlock2d(out_channels + skip_channels, out_channels, conv5_mode=conv5_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class DMCNetPolyp2d(nn.Module):
    """2D DMC-Net for RGB binary polyp segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256, 512, 512),
        conv5_mode: str = "depthwise",
    ):
        super().__init__()
        if len(features) != 6:
            raise ValueError("DMCNetPolyp2d expects 6 feature levels, e.g. (32, 64, 128, 256, 512, 512).")

        self.enc1 = DMCBlock2d(in_channels, features[0], conv5_mode=conv5_mode)
        self.enc2 = DMCBlock2d(features[0], features[1], conv5_mode=conv5_mode)
        self.enc3 = DMCBlock2d(features[1], features[2], conv5_mode=conv5_mode)
        self.enc4 = DMCBlock2d(features[2], features[3], conv5_mode=conv5_mode)
        self.enc5 = DMCBlock2d(features[3], features[4], conv5_mode=conv5_mode)
        self.bottleneck = DMCBlock2d(features[4], features[5], conv5_mode=conv5_mode)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec5 = UpBlock2d(features[5], features[4], features[4], conv5_mode=conv5_mode)
        self.dec4 = UpBlock2d(features[4], features[3], features[3], conv5_mode=conv5_mode)
        self.dec3 = UpBlock2d(features[3], features[2], features[2], conv5_mode=conv5_mode)
        self.dec2 = UpBlock2d(features[2], features[1], features[1], conv5_mode=conv5_mode)
        self.dec1 = UpBlock2d(features[1], features[0], features[0], conv5_mode=conv5_mode)
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
