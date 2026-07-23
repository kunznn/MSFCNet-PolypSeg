import torch


def _to_probabilities(preds: torch.Tensor) -> torch.Tensor:
    if preds.min() < 0 or preds.max() > 1:
        return torch.sigmoid(preds)
    return preds


@torch.no_grad()
def binary_segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    probs = _to_probabilities(preds).float()
    targets = (targets > 0.5).float()
    binary = (probs >= threshold).float()

    dims = tuple(range(1, binary.ndim))
    tp = (binary * targets).sum(dim=dims)
    fp = (binary * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - binary) * targets).sum(dim=dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    mae = torch.abs(probs - targets).mean(dim=dims)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "mae": mae.mean().item(),
    }
