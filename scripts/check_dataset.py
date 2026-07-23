import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEST_COUNTS = {
    "Kvasir": 100,
    "CVC-ClinicDB": 62,
    "CVC-300": 60,
    "CVC-ColonDB": 380,
    "ETIS-LaribPolypDB": 196,
}


def list_pngs(path: Path):
    return sorted(path.glob("*.png")) if path.exists() else []


def inspect_split(name, image_dir, mask_dir, strict: bool = True):
    images = list_pngs(image_dir)
    masks = list_pngs(mask_dir)
    if not image_dir.exists() or not mask_dir.exists():
        message = f"{name}: missing image or mask folder"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return images, masks, False
    if not images or not masks:
        print(f"WARNING: {name}: empty split ({len(images)} images, {len(masks)} masks)")
        return images, masks, len(images) == len(masks)
    if len(images) != len(masks):
        message = f"{name}: image/mask count mismatch ({len(images)} images, {len(masks)} masks)"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return images, masks, False
    missing = [p.name for p in images if not (mask_dir / p.name).exists()]
    if missing:
        message = f"{name}: masks missing for {missing[:10]}"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return images, masks, False
    return images, masks, True


def mask_is_binary(mask_path: Path):
    values = np.unique(np.asarray(Image.open(mask_path).convert("L")))
    return set(values.tolist()).issubset({0, 255}), values


def save_random_visualization(pairs, output_path: Path, n: int = 10):
    if not pairs:
        return
    sample = random.sample(pairs, k=min(n, len(pairs)))
    fig, axes = plt.subplots(len(sample), 2, figsize=(6, 3 * len(sample)))
    if len(sample) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row, (image_path, mask_path) in enumerate(sample):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(image_path.name)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("mask")
        axes[row, 1].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def check_masks(mask_paths):
    all_binary = True
    bad_masks = []
    for mask_path in mask_paths:
        is_binary, values = mask_is_binary(mask_path)
        if not is_binary:
            all_binary = False
            bad_masks.append((mask_path.name, values[:10].tolist()))
    return all_binary, bad_masks


def main():
    processed = ROOT / "data/processed"
    if not processed.exists():
        print("Processed data does not exist. Please run: python scripts/prepare_data.py --seed 42")
        sys.exit(1)

    fixed_splits = {
        "train": (processed / "train/images", processed / "train/masks"),
        "val": (processed / "val/images", processed / "val/masks"),
    }
    all_pairs = []
    all_sizes = []

    print("Train/Val")
    for split_name, (image_dir, mask_dir) in fixed_splits.items():
        images, masks, matched = inspect_split(split_name, image_dir, mask_dir, strict=True)
        all_binary, bad_masks = check_masks(masks)
        print(
            f"  {split_name}: images={len(images)}, masks={len(masks)}, "
            f"matched={matched}, binary_masks={all_binary}"
        )
        if bad_masks:
            print(f"    first non-binary masks: {bad_masks[:5]}")
        for image_path in images:
            mask_path = mask_dir / image_path.name
            all_pairs.append((image_path, mask_path))
            all_sizes.append(Image.open(image_path).size)

    print("Test Sets")
    test_root = processed / "test"
    for dataset_name in sorted(EXPECTED_TEST_COUNTS):
        image_dir = test_root / dataset_name / "images"
        mask_dir = test_root / dataset_name / "masks"
        images, masks, matched = inspect_split(dataset_name, image_dir, mask_dir, strict=False)
        all_binary, bad_masks = check_masks(masks)
        expected = EXPECTED_TEST_COUNTS[dataset_name]
        count_note = "OK" if len(images) == expected else f"expected {expected}"
        print(
            f"  {dataset_name}: images={len(images)}, masks={len(masks)}, "
            f"matched={matched}, binary_masks={all_binary}, {count_note}"
        )
        if bad_masks:
            print(f"    first non-binary masks: {bad_masks[:5]}")
        for image_path in images:
            mask_path = mask_dir / image_path.name
            if mask_path.exists():
                all_pairs.append((image_path, mask_path))
                all_sizes.append(Image.open(image_path).size)

    if all_pairs:
        save_random_visualization(all_pairs, ROOT / "results/visualizations/dataset_check.png", n=10)

    if all_sizes:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        print(f"image size range: width {min(widths)}-{max(widths)}, height {min(heights)}-{max(heights)}")
    print("Saved visualization: results/visualizations/dataset_check.png")


if __name__ == "__main__":
    main()
