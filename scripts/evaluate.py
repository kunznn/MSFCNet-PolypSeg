import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.polyp_dataset import PolypDataset, build_eval_transform
from src.metrics.segmentation_metrics import binary_segmentation_metrics
from src.models.factory import build_model
from src.utils.visualization import denormalize_image, overlay_mask

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_prediction(prob, output_path: Path, threshold: float):
    pred = (prob >= threshold).astype(np.uint8) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(output_path)


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary_mask
    if cv2 is None:
        return binary_mask
    mask = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned


def predict_probabilities(model, images, tta: bool):
    def final_output(outputs):
        return outputs[-1] if isinstance(outputs, (tuple, list)) else outputs

    if not tta:
        return torch.sigmoid(final_output(model(images)))

    probs = torch.sigmoid(final_output(model(images)))

    h_images = torch.flip(images, dims=[3])
    h_probs = torch.flip(torch.sigmoid(final_output(model(h_images))), dims=[3])

    v_images = torch.flip(images, dims=[2])
    v_probs = torch.flip(torch.sigmoid(final_output(model(v_images))), dims=[2])

    hv_images = torch.flip(images, dims=[2, 3])
    hv_probs = torch.flip(torch.sigmoid(final_output(model(hv_images))), dims=[2, 3])

    return (probs + h_probs + v_probs + hv_probs) / 4.0


def metrics_from_probs(probs, masks, threshold: float, min_area: int, eps: float = 1e-7):
    probs_np = probs.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    values = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "weighted_fmeasure": [],
        "s_measure": [],
        "e_measure_max": [],
        "mae": [],
    }

    for prob, mask in zip(probs_np[:, 0], masks_np[:, 0]):
        target = (mask > 0.5).astype(np.float32)
        pred = (prob >= threshold).astype(np.uint8)
        pred = remove_small_components(pred, min_area).astype(np.float32)

        tp = float((pred * target).sum())
        fp = float((pred * (1.0 - target)).sum())
        fn = float(((1.0 - pred) * target).sum())

        values["dice"].append((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        values["iou"].append((tp + eps) / (tp + fp + fn + eps))
        values["precision"].append((tp + eps) / (tp + fp + eps))
        values["recall"].append((tp + eps) / (tp + fn + eps))
        values["weighted_fmeasure"].append(weighted_fmeasure(prob, target, eps=eps))
        values["s_measure"].append(s_measure(prob, target, eps=eps))
        values["e_measure_max"].append(max_e_measure(prob, target, eps=eps))
        values["mae"].append(float(np.abs(prob - target).mean()))

    return {key: float(np.mean(value)) for key, value in values.items()}


def weighted_fmeasure(prob: np.ndarray, target: np.ndarray, beta2: float = 0.3, eps: float = 1e-7) -> float:
    """Weighted F-measure for foreground maps, following the common saliency evaluation protocol."""
    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
    gt = (target > 0.5)
    if gt.sum() == 0:
        return 0.0
    if ndimage is None or cv2 is None:
        pred = (prob >= 0.5).astype(np.float32)
        tp = float((pred * gt).sum())
        fp = float((pred * (~gt)).sum())
        fn = float(((1.0 - pred) * gt).sum())
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        return float((1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps))

    error = np.abs(prob - gt.astype(np.float32))
    distance, indices = ndimage.distance_transform_edt(~gt, return_indices=True)
    error_nearest = error[tuple(indices)]
    error_propagated = error.copy()
    error_propagated[~gt] = error_nearest[~gt]
    error_smoothed = cv2.GaussianBlur(error_propagated, (7, 7), 5)

    min_error = error.copy()
    fg_smaller = gt & (error_smoothed < error)
    min_error[fg_smaller] = error_smoothed[fg_smaller]

    weight = np.ones_like(prob, dtype=np.float32)
    weight[~gt] = 2.0 - np.exp(np.log(0.5) / 5.0 * distance[~gt])
    weighted_error = min_error * weight

    tp_weighted = float(gt.sum() - weighted_error[gt].sum())
    fp_weighted = float(weighted_error[~gt].sum())
    recall_weighted = 1.0 - float(weighted_error[gt].mean())
    precision_weighted = tp_weighted / (tp_weighted + fp_weighted + eps)
    score = (1.0 + beta2) * precision_weighted * recall_weighted / (beta2 * precision_weighted + recall_weighted + eps)
    return float(np.clip(score, 0.0, 1.0))


def s_measure(prob: np.ndarray, target: np.ndarray, alpha: float = 0.5, eps: float = 1e-7) -> float:
    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
    gt = (target > 0.5).astype(np.float32)
    gt_mean = float(gt.mean())
    if gt_mean == 0.0:
        return float(1.0 - prob.mean())
    if gt_mean == 1.0:
        return float(prob.mean())
    return float(alpha * _s_object(prob, gt, eps) + (1.0 - alpha) * _s_region(prob, gt, eps))


def _object_score(values: np.ndarray, eps: float) -> float:
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    std = float(values.std())
    return (2.0 * mean) / (mean * mean + 1.0 + std + eps)


def _s_object(prob: np.ndarray, gt: np.ndarray, eps: float) -> float:
    fg = prob[gt == 1]
    bg = 1.0 - prob[gt == 0]
    gt_mean = float(gt.mean())
    return gt_mean * _object_score(fg, eps) + (1.0 - gt_mean) * _object_score(bg, eps)


def _centroid(gt: np.ndarray):
    h, w = gt.shape
    area = gt.sum()
    if area == 0:
        return w // 2, h // 2
    y_idx, x_idx = np.indices((h, w))
    x = int(np.round((x_idx * gt).sum() / area))
    y = int(np.round((y_idx * gt).sum() / area))
    return min(max(x, 1), w - 1), min(max(y, 1), h - 1)


def _ssim(pred: np.ndarray, gt: np.ndarray, eps: float) -> float:
    if pred.size == 0:
        return 0.0
    pred_mean = float(pred.mean())
    gt_mean = float(gt.mean())
    pred_var = float(((pred - pred_mean) ** 2).mean())
    gt_var = float(((gt - gt_mean) ** 2).mean())
    covariance = float(((pred - pred_mean) * (gt - gt_mean)).mean())
    numerator = 4.0 * pred_mean * gt_mean * covariance
    denominator = (pred_mean * pred_mean + gt_mean * gt_mean) * (pred_var + gt_var) + eps
    return numerator / denominator if denominator != 0 else 0.0


def _s_region(prob: np.ndarray, gt: np.ndarray, eps: float) -> float:
    h, w = gt.shape
    x, y = _centroid(gt)
    regions = [
        (slice(0, y), slice(0, x)),
        (slice(0, y), slice(x, w)),
        (slice(y, h), slice(0, x)),
        (slice(y, h), slice(x, w)),
    ]
    weights = [
        (x * y) / (w * h),
        ((w - x) * y) / (w * h),
        (x * (h - y)) / (w * h),
        ((w - x) * (h - y)) / (w * h),
    ]
    return sum(weight * _ssim(prob[r], gt[r], eps) for weight, r in zip(weights, regions))


def max_e_measure(prob: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
    gt = (target > 0.5).astype(np.float32)
    gt_mean = float(gt.mean())
    if gt_mean == 0.0:
        return float(1.0 - prob.mean())
    if gt_mean == 1.0:
        return float(prob.mean())

    scores = []
    for threshold in np.linspace(0.0, 1.0, 256):
        binary = (prob >= threshold).astype(np.float32)
        scores.append(_e_measure(binary, gt, eps))
    return float(max(scores))


def _e_measure(pred: np.ndarray, gt: np.ndarray, eps: float) -> float:
    pred_centered = pred - pred.mean()
    gt_centered = gt - gt.mean()
    alignment = (2.0 * pred_centered * gt_centered) / (pred_centered ** 2 + gt_centered ** 2 + eps)
    enhanced = ((alignment + 1.0) ** 2) / 4.0
    return float(enhanced.mean())


def discover_test_datasets(test_root: Path):
    datasets = []
    if not test_root.exists():
        return datasets
    for dataset_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        image_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            print(f"WARNING: skipping {dataset_dir.name}; missing images/ or masks/")
            continue
        if not list(image_dir.glob("*.png")):
            print(f"WARNING: skipping {dataset_dir.name}; no PNG images found")
            continue
        datasets.append((dataset_dir.name, image_dir, mask_dir))
    return datasets


def evaluate_dataset(model, loader, device, dataset_name, output_dir, threshold, tta, min_area):
    model.eval()
    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "weighted_fmeasure": 0.0,
        "s_measure": 0.0,
        "e_measure_max": 0.0,
        "mae": 0.0,
    }
    total_samples = 0
    qualitative = []

    with torch.no_grad():
        for images, masks, names in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            probs = predict_probabilities(model, images, tta=tta)
            metrics = metrics_from_probs(probs, masks, threshold=threshold, min_area=min_area)
            batch_size = images.shape[0]
            for key in totals:
                totals[key] += metrics[key] * batch_size
            total_samples += batch_size

            probs_np = probs.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            for i, name in enumerate(names):
                prob = probs_np[i, 0]
                pred = remove_small_components((prob >= threshold).astype(np.uint8), min_area)
                output_path = output_dir / name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pred.astype(np.uint8) * 255).save(output_path)
                if len(qualitative) < 6:
                    qualitative.append(
                        {
                            "image": denormalize_image(images[i]),
                            "gt": masks_np[i, 0],
                            "pred": pred.astype(np.float32),
                        }
                    )

    averaged = {key: value / max(1, total_samples) for key, value in totals.items()}
    return averaged, qualitative


def save_qualitative(samples, output_path: Path):
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        image = sample["image"]
        gt = sample["gt"]
        pred = sample["pred"]
        overlay = overlay_mask(image, pred)
        panels = [(image, "Input"), (gt, "GT"), (pred, "Prediction"), (overlay, "Overlay")]
        for col, (data, title) in enumerate(panels):
            if data.ndim == 2:
                axes[row, col].imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                axes[row, col].imshow(data)
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--config", default="configs/unet_kvasir_cvc.yaml")
    parser.add_argument("--threshold", type=float, default=None, help="Override config threshold, e.g. 0.45")
    parser.add_argument("--tta", action="store_true", help="Use flip test-time augmentation")
    parser.add_argument("--min_area", type=int, default=0, help="Remove predicted connected components smaller than this area")
    parser.add_argument(
        "--postprocess_config",
        default=None,
        help="Load validation-selected threshold, min_area, and tta from JSON.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/unet_test_results.csv",
        help="CSV path for saving evaluation results.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional model tag used to save predictions/visualizations without overwriting other models.",
    )
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train first, then evaluate the best checkpoint.")
        sys.exit(1)

    config = load_config(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_kwargs = {
        "in_channels": config["model"].get("in_channels", 3),
        "out_channels": config["model"].get("out_channels", 1),
    }
    if "features" in config["model"]:
        model_kwargs["features"] = tuple(config["model"]["features"])
    if "conv5_mode" in config["model"]:
        model_kwargs["conv5_mode"] = config["model"]["conv5_mode"]
    if "channel" in config["model"]:
        model_kwargs["channel"] = config["model"]["channel"]
    if "pretrained" in config["model"]:
        model_kwargs["pretrained"] = config["model"]["pretrained"]
    if "use_msm" in config["model"]:
        model_kwargs["use_msm"] = config["model"]["use_msm"]
    if "use_bcf" in config["model"]:
        model_kwargs["use_bcf"] = config["model"]["use_bcf"]
    if "decoder_channels" in config["model"]:
        model_kwargs["decoder_channels"] = tuple(config["model"]["decoder_channels"])
    model = build_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image_size = config.get("image_size", 352)
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)
    threshold = args.threshold if args.threshold is not None else config.get("threshold", 0.5)
    tta = args.tta
    min_area = args.min_area
    output_tag = args.tag

    if args.postprocess_config is not None:
        postprocess_path = ROOT / args.postprocess_config
        with open(postprocess_path, "r", encoding="utf-8") as f:
            postprocess_config = json.load(f)
        threshold = float(postprocess_config["threshold"])
        min_area = int(postprocess_config.get("min_area", 0))
        tta = bool(postprocess_config.get("tta", False))

    datasets = discover_test_datasets(ROOT / "data/processed/test")
    if not datasets:
        print("No test datasets found under data/processed/test")
        sys.exit(1)

    rows = []
    for dataset_name, image_dir, mask_dir in datasets:
        dataset = PolypDataset(image_dir, mask_dir, image_size, transform=build_eval_transform(image_size))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        if output_tag:
            pred_dir = ROOT / "results" / "predictions" / output_tag / dataset_name
            qualitative_path = ROOT / "results" / "visualizations" / output_tag / f"qualitative_{dataset_name}.png"
        else:
            pred_dir = ROOT / "results" / "predictions" / dataset_name
            qualitative_path = ROOT / "results" / "visualizations" / f"unet_qualitative_{dataset_name}.png"
        metrics, qualitative = evaluate_dataset(model, loader, device, dataset_name, pred_dir, threshold, tta, min_area)
        save_qualitative(qualitative, qualitative_path)
        rows.append(
            {
                "dataset": dataset_name,
                "num_images": len(dataset),
                "threshold": threshold,
                "min_area": min_area,
                "tta": tta,
                **metrics,
            }
        )
        print(
            f"{dataset_name}: n={len(dataset)}, Dice {metrics['dice']:.4f}, IoU {metrics['iou']:.4f}, "
            f"Fbw {metrics['weighted_fmeasure']:.4f}, S {metrics['s_measure']:.4f}, "
            f"Emax {metrics['e_measure_max']:.4f}, MAE {metrics['mae']:.4f}"
        )

    output_csv = ROOT / args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "num_images",
                "threshold",
                "min_area",
                "tta",
                "dice",
                "iou",
                "precision",
                "recall",
                "weighted_fmeasure",
                "s_measure",
                "e_measure_max",
                "mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Threshold: {threshold:.3f}")
    print(f"TTA: {tta}, min_area: {min_area}")
    if output_tag:
        print(f"Tag: {output_tag}")
    print(f"Saved test results: {output_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
