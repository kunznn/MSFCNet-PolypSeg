from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None


def build_train_augmentation(image_size: int, config: dict | None = None):
    if A is None:
        raise ImportError("albumentations is required for augmentation. Install it with: pip install albumentations")

    config = config or {}
    scale_limit = config.get("scale_limit", 0.2)
    return A.Compose(
        [
            A.HorizontalFlip(p=config.get("flip_p", 0.5)),
            A.VerticalFlip(p=config.get("flip_p", 0.5)),
            A.Rotate(
                limit=config.get("rotation_limit", 30),
                border_mode=0,
                p=config.get("rotation_p", 0.3),
            ),
            A.Affine(scale=(1.0 - scale_limit, 1.0 + scale_limit), p=config.get("scale_p", 0.25)),
            A.ElasticTransform(alpha=50, sigma=6, p=config.get("elastic_p", 0.1)),
            A.Resize(image_size, image_size),
            A.GaussNoise(p=config.get("gaussian_noise_p", 0.15)),
            A.GaussianBlur(blur_limit=(3, 5), p=config.get("gaussian_blur_p", 0.2)),
            A.RandomBrightnessContrast(p=config.get("brightness_contrast_p", 0.25)),
        ]
    )


def build_eval_transform(image_size: int):
    if A is None:
        return None
    return A.Compose([A.Resize(image_size, image_size)])


class PolypDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size: int = 352, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        self.mask_paths = [self.masks_dir / f"{p.stem}.png" for p in self.image_paths]

        missing = [str(p) for p in self.mask_paths if not p.exists()]
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(f"Missing masks for {len(missing)} images. First missing masks:\n{preview}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.asarray(image)
        mask_np = np.asarray(mask)

        if self.transform is not None:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented["image"]
            mask_np = augmented["mask"]
        else:
            size = (self.image_size, self.image_size)
            image_np = np.asarray(image.resize(size, Image.BILINEAR))
            mask_np = np.asarray(mask.resize(size, Image.NEAREST))

        image_np = image_np.astype(np.float32) / 255.0
        mask_np = (mask_np.astype(np.uint8) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return image_tensor, mask_tensor, image_path.name
