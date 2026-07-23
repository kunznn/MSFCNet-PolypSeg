from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScalePerceptionModule(nn.Module):
    """Light ASPP-style multi-scale context module."""

    def __init__(self, channels: int):
        super().__init__()
        branch_channels = max(channels // 4, 16)
        self.branch1 = ConvBNReLU(channels, branch_channels, kernel_size=1, padding=0)
        self.branch3 = DepthwiseSeparableConv(channels, branch_channels, kernel_size=3, dilation=1)
        self.branch5 = DepthwiseSeparableConv(channels, branch_channels, kernel_size=5, dilation=1)
        self.branch_dilated = DepthwiseSeparableConv(channels, branch_channels, kernel_size=3, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_dilated(x)],
            dim=1,
        )
        return x + self.fuse(multi_scale)


class CalibrationFusionBlock(nn.Module):
    """Semantic-guided skip calibration before decoder fusion."""

    def __init__(self, high_channels: int, skip_channels: int, out_channels: int, use_bcf: bool = True):
        super().__init__()
        self.use_bcf = use_bcf
        self.high_proj = ConvBNReLU(high_channels, out_channels, kernel_size=1, padding=0)
        self.skip_proj = ConvBNReLU(skip_channels, out_channels, kernel_size=1, padding=0)
        if use_bcf:
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        self.fuse = DoubleConv(out_channels * 2, out_channels)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        high = self.high_proj(high)
        skip = self.skip_proj(skip)
        if self.use_bcf:
            skip = skip * self.attention(high) + skip
        return self.fuse(torch.cat([skip, high], dim=1))


class PolypMFCNet(nn.Module):
    """ResNet34 encoder + lightweight decoder with optional MSM and FCF."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        use_msm: bool = True,
        use_bcf: bool = True,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 64),
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("PolypMFCNet currently expects RGB input with in_channels=3.")

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.use_msm = use_msm
        if use_msm:
            self.msm2 = MultiScalePerceptionModule(128)
            self.msm3 = MultiScalePerceptionModule(256)
            self.msm4 = MultiScalePerceptionModule(512)

        d4, d3, d2, d1 = decoder_channels
        self.dec4 = CalibrationFusionBlock(512, 256, d4, use_bcf=use_bcf)
        self.dec3 = CalibrationFusionBlock(d4, 128, d3, use_bcf=use_bcf)
        self.dec2 = CalibrationFusionBlock(d3, 64, d2, use_bcf=use_bcf)
        self.dec1 = CalibrationFusionBlock(d2, 64, d1, use_bcf=use_bcf)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(d1, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.use_msm:
            x2 = self.msm2(x2)
            x3 = self.msm3(x3)
            x4 = self.msm4(x4)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        logits = self.final(d1)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
