import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class MobileNetV2UNet(nn.Module):
    """MobileNetV2 encoder with a lightweight U-Net style decoder for binary polyp segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("MobileNetV2UNet currently expects RGB input with in_channels=3.")

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = mobilenet_v2(weights=weights).features
        self.stem = encoder[:2]      # 16, H/2
        self.enc2 = encoder[2:4]     # 24, H/4
        self.enc3 = encoder[4:7]     # 32, H/8
        self.enc4 = encoder[7:14]    # 96, H/16
        self.enc5 = encoder[14:]     # 1280, H/32

        d1, d2, d3, d4 = decoder_channels
        self.up4 = UpBlock(1280, 96, d1)
        self.up3 = UpBlock(d1, 32, d2)
        self.up2 = UpBlock(d2, 24, d3)
        self.up1 = UpBlock(d3, 16, d4)
        self.head = nn.Conv2d(d4, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x1 = self.stem(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
