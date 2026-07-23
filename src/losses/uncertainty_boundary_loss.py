from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dice_bce_loss import DiceLoss


class UncertaintyBoundaryBCEDiceLoss(nn.Module):
    """BCE-Dice loss with optional uncertainty-guided boundary weighting.

    UBL only changes the training loss. It does not add inference parameters
    or FLOPs to the segmentation model.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_lambda: float = 1.0,
        boundary_kernel_size: int = 5,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_lambda = boundary_lambda
        self.boundary_kernel_size = boundary_kernel_size
        self.eps = eps
        self.dice = DiceLoss(eps=eps)

    def _boundary_map(self, targets: torch.Tensor) -> torch.Tensor:
        kernel = self.boundary_kernel_size
        padding = kernel // 2
        dilated = F.max_pool2d(targets, kernel_size=kernel, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool2d(1.0 - targets, kernel_size=kernel, stride=1, padding=padding)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits).detach()
        uncertainty = 1.0 - torch.abs(2.0 * probs - 1.0)
        boundary = self._boundary_map(targets)
        weights = 1.0 + self.boundary_lambda * uncertainty * boundary
        weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(self.eps)
        return self.bce_weight * weighted_bce + self.dice_weight * self.dice(logits, targets)
