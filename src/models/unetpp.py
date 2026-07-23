from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class UNetPlusPlusDoubleConv2d(nn.Module):
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


class UNetPlusPlus2D(nn.Module):
    """UNet++ with nested dense skip connections for 2D binary segmentation.

    The model returns single-channel logits by default. Sigmoid should be applied
    in the loss or evaluation code, not inside the model forward pass.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256, 512, 512),
    ) -> None:
        super().__init__()
        if len(features) != 6:
            raise ValueError("UNetPlusPlus2D expects 6 feature levels, e.g. (32, 64, 128, 256, 512, 512).")

        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoders = nn.ModuleList()
        self.encoders.append(UNetPlusPlusDoubleConv2d(in_channels, features[0]))
        for level in range(1, len(features)):
            self.encoders.append(UNetPlusPlusDoubleConv2d(features[level - 1], features[level]))

        self.nested_convs = nn.ModuleDict()
        depth = len(features) - 1
        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                in_ch = j * features[i] + features[i + 1]
                out_ch = features[i]
                self.nested_convs[f"x{i}_{j}"] = UNetPlusPlusDoubleConv2d(in_ch, out_ch)

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _upsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes: dict[tuple[int, int], torch.Tensor] = {}
        nodes[(0, 0)] = self.encoders[0](x)
        for level in range(1, len(self.features)):
            nodes[(level, 0)] = self.encoders[level](self.pool(nodes[(level - 1, 0)]))

        depth = len(self.features) - 1
        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                same_level = [nodes[(i, k)] for k in range(j)]
                up = self._upsample_like(nodes[(i + 1, j - 1)], nodes[(i, 0)])
                nodes[(i, j)] = self.nested_convs[f"x{i}_{j}"](torch.cat([*same_level, up], dim=1))

        return self.head(nodes[(0, depth)])
