import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.polyp_dataset import PolypDataset, build_eval_transform, build_train_augmentation
from src.losses.dice_bce_loss import BCEDiceLoss
from src.metrics.segmentation_metrics import binary_segmentation_metrics
from src.models.pranet import PraNet
from src.utils.seed import set_seed
from src.utils.visualization import save_training_curve


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_processed_data(config):
    required = {
        "train images": config["data"]["train_images"],
        "train masks": config["data"]["train_masks"],
        "val images": config["data"]["val_images"],
        "val masks": config["data"]["val_masks"],
    }
    missing = [p for p in required.values() if not (ROOT / p).exists()]
    empty = [name for name, path in required.items() if (ROOT / path).exists() and not list((ROOT / path).glob("*.png"))]
    if missing or empty:
        print("Processed data is missing or empty. Please run: python scripts/prepare_data.py --seed 42")
        if missing:
            print(f"Missing paths: {missing}")
        if empty:
            print(f"Empty folders: {empty}")
        sys.exit(1)


def final_output(outputs):
    return outputs[-1] if isinstance(outputs, (tuple, list)) else outputs


def multi_output_loss(outputs, masks, criterion):
    if not isinstance(outputs, (tuple, list)):
        return criterion(outputs, masks)
    return sum(criterion(output, masks) for output in outputs) / len(outputs)


def tensor_range(name, tensor):
    finite = torch.isfinite(tensor)
    if not finite.all():
        print(f"{name}: contains non-finite values")
    clean = tensor[finite]
    if clean.numel() == 0:
        print(f"{name}: no finite values")
        return
    print(f"{name}: min {clean.min().item():.6f}, max {clean.max().item():.6f}, mean {clean.mean().item():.6f}")


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


def evaluate(model, loader, device, threshold):
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "mae": 0.0}
    batches = 0
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = final_output(model(images))
            metrics = binary_segmentation_metrics(logits, masks, threshold=threshold)
            for key in totals:
                totals[key] += metrics[key]
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b04_pranet_kvasir_cvc.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    validate_processed_data(config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config.get("image_size", 352)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    augmentation_config = config.get("augmentation", {})
    train_transform = build_train_augmentation(image_size, augmentation_config) if augmentation_config.get("enabled", False) else None
    val_transform = build_eval_transform(image_size)

    train_dataset = PolypDataset(ROOT / config["data"]["train_images"], ROOT / config["data"]["train_masks"], image_size, transform=train_transform)
    val_dataset = PolypDataset(ROOT / config["data"]["val_images"], ROOT / config["data"]["val_masks"], image_size, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = PraNet(
        in_channels=config["model"].get("in_channels", 3),
        out_channels=config["model"].get("out_channels", 1),
        channel=config["model"].get("channel", 32),
        pretrained=config["model"].get("pretrained", False),
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-5))
    grad_clip_norm = config.get("grad_clip_norm", 1.0)
    debug_first_batch = config.get("debug_first_batch", True)

    epochs = config.get("epochs", 300)
    threshold = config.get("threshold", 0.5)
    early_stopping_patience = config.get("early_stopping_patience")
    scheduler = None
    if str(config.get("scheduler", "")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.get("min_lr", 1e-6))

    best_dice = -1.0
    epochs_without_improvement = 0
    logs = []
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "results/visualizations").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, masks, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            if debug_first_batch and epoch == 1 and step == 0:
                print("PraNet debug first batch")
                tensor_range("image range", images)
                tensor_range("mask range", masks)
                if isinstance(outputs, (tuple, list)):
                    for output_index, output in enumerate(outputs):
                        tensor_range(f"output[{output_index}] logits range", output)
                else:
                    tensor_range("output logits range", outputs)
            if isinstance(outputs, (tuple, list)):
                for output_index, output in enumerate(outputs):
                    assert_finite(f"output[{output_index}]", output)
            else:
                assert_finite("output", outputs)
            loss = multi_output_loss(outputs, masks, criterion)
            if debug_first_batch and epoch == 1 and step == 0:
                print(f"first batch loss: {loss.item():.6f}")
            assert_finite("loss", loss)
            loss.backward()
            if grad_clip_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Gradient norm is NaN or Inf: {grad_norm}")
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device, threshold)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_mae": val_metrics["mae"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        logs.append(row)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val Dice {val_metrics['dice']:.4f} | val IoU {val_metrics['iou']:.4f} | val MAE {val_metrics['mae']:.4f}")

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_dice": best_dice, "config": config}, ROOT / "checkpoints/pranet_best.pth")
        else:
            epochs_without_improvement += 1

        with open(ROOT / "logs/pranet_train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            writer.writeheader()
            writer.writerows(logs)
        save_training_curve(pd.DataFrame(logs), ROOT / "results/visualizations/pranet_training_curve.png")

        if scheduler is not None:
            scheduler.step()
        if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Val Dice did not improve for {early_stopping_patience} epochs.")
            break

    print(f"Training complete. Best val Dice: {best_dice:.4f}")
    print("Best checkpoint: checkpoints/pranet_best.pth")


if __name__ == "__main__":
    main()
