from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def denormalize_image(tensor):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def save_training_curve(log_df, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(log_df["epoch"], log_df["val_dice"], label="Val Dice", color="tab:green")
    ax2.set_ylabel("Val Dice", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def overlay_mask(image, mask, color=(1.0, 0.0, 0.0), alpha=0.35):
    image = np.asarray(image).copy()
    mask = np.asarray(mask) > 0
    color_arr = np.array(color, dtype=np.float32)
    image[mask] = (1 - alpha) * image[mask] + alpha * color_arr
    return image
