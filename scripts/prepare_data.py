import argparse
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXTERNAL_TESTSETS = {
    "CVC-300": {"prefix": "cvc300", "split": "test_cvc300.txt"},
    "CVC-ColonDB": {"prefix": "colondb", "split": "test_colondb.txt"},
    "ETIS-LaribPolypDB": {"prefix": "etis", "split": "test_etis.txt"},
}


def list_dirs(path: Path):
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_dir()]


def find_named_dir(root: Path, keywords):
    dirs = list_dirs(root)
    scored = []
    for path in dirs:
        normalized = path.name.lower().replace("_", " ").replace("-", " ")
        if any(keyword in normalized for keyword in keywords):
            files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if files:
                scored.append((len(files), path))
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1] if scored else None


def find_image_mask_dirs(dataset_root: Path):
    image_names = ["images", "image", "imgs", "img", "original"]
    mask_names = ["masks", "mask", "gt", "ground truth", "groundtruth", "labels", "label"]
    dirs = [dataset_root, *list_dirs(dataset_root)]

    def normalize(name):
        return name.lower().replace("_", " ").replace("-", " ")

    image_candidates = []
    mask_candidates = []
    for path in dirs:
        if not path.is_dir():
            continue
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue
        normalized = normalize(path.name)
        if normalized in image_names or any(name == normalized for name in image_names):
            image_candidates.append((len(files), path))
        if normalized in mask_names or any(name == normalized for name in mask_names):
            mask_candidates.append((len(files), path))

    if not image_candidates:
        image_dir = find_named_dir(dataset_root, image_names)
    else:
        image_candidates.sort(reverse=True, key=lambda item: item[0])
        image_dir = image_candidates[0][1]

    if not mask_candidates:
        mask_dir = find_named_dir(dataset_root, mask_names)
    else:
        mask_candidates.sort(reverse=True, key=lambda item: item[0])
        mask_dir = mask_candidates[0][1]

    return image_dir, mask_dir


def find_kvasir_dirs(root: Path):
    canonical_images = root / "data" / "raw" / "Kvasir-SEG" / "images"
    canonical_masks = root / "data" / "raw" / "Kvasir-SEG" / "masks"
    if canonical_images.exists() and canonical_masks.exists():
        return canonical_images, canonical_masks

    candidates = [
        root / "Kvasir-SEG-main",
        root / "Kvasir-SEG",
        root / "kvasir-seg",
        root / "archive" / "Kvasir-SEG",
        root / "archive" / "Kvasir-SEG" / "Kvasir-SEG",
    ]
    for base in candidates:
        if not base.exists():
            continue
        image_dir = find_named_dir(base, ["image", "images"])
        mask_dir = find_named_dir(base, ["mask", "masks", "label", "labels"])
        if image_dir and mask_dir and image_dir != mask_dir:
            return image_dir, mask_dir
    return None, None


def find_cvc_dirs(root: Path):
    canonical_images = root / "data" / "raw" / "CVC-ClinicDB" / "images"
    canonical_masks = root / "data" / "raw" / "CVC-ClinicDB" / "masks"
    if canonical_images.exists() and canonical_masks.exists():
        return canonical_images, canonical_masks

    base = root / "archive"
    if not base.exists():
        return None, None

    image_candidates = [
        base / "PNG" / "Original",
        base / "TIF" / "Original",
        base / "images",
        base / "Images",
        base / "Original",
        base / "original",
    ]
    mask_candidates = [
        base / "PNG" / "Ground Truth",
        base / "TIF" / "Ground Truth",
        base / "masks",
        base / "Masks",
        base / "ground truth",
        base / "Ground Truth",
    ]

    image_dir = next((p for p in image_candidates if p.exists()), None)
    mask_dir = next((p for p in mask_candidates if p.exists()), None)
    if image_dir and mask_dir:
        return image_dir, mask_dir

    return find_named_dir(base, ["image", "original"]), find_named_dir(base, ["mask", "ground truth", "label"])


def collect_pairs(image_dir: Path, mask_dir: Path):
    image_files = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    pairs = [(image_path, mask_map[image_path.stem]) for image_path in image_files if image_path.stem in mask_map]
    return pairs


def clear_processed_dirs():
    targets = [
        ROOT / "data/processed/train/images",
        ROOT / "data/processed/train/masks",
        ROOT / "data/processed/val/images",
        ROOT / "data/processed/val/masks",
        ROOT / "data/processed/test/Kvasir/images",
        ROOT / "data/processed/test/Kvasir/masks",
        ROOT / "data/processed/test/CVC-ClinicDB/images",
        ROOT / "data/processed/test/CVC-ClinicDB/masks",
        ROOT / "data/splits",
    ]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for file_path in target.glob("*"):
            if file_path.is_file():
                file_path.unlink()


def save_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask_np = np.asarray(mask)
    mask_bin = np.where(mask_np > 0, 255, 0).astype(np.uint8)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image)
    Image.fromarray(mask_bin).save(out_mask)


def split_dataset(pairs, seed: int, test_ratio: float = 0.1):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n_test = max(1, int(math.ceil(len(pairs) * test_ratio))) if len(pairs) > 1 else len(pairs)
    return pairs[n_test:], pairs[:n_test]


def write_split_file(path: Path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def print_detection_help():
    print("\nCould not automatically detect required image/mask folders.")
    print("Detected folders under archive:")
    for path in list_dirs(ROOT / "archive")[:80]:
        print(f"  - {path.relative_to(ROOT)}")
    print("Detected folders under Kvasir-SEG-main:")
    for path in list_dirs(ROOT / "Kvasir-SEG-main")[:80]:
        print(f"  - {path.relative_to(ROOT)}")
    print("\nPlease check folder names or edit find_kvasir_dirs/find_cvc_dirs in scripts/prepare_data.py.")


def export_dataset(items, prefix, image_out_dir, mask_out_dir):
    names = []
    for index, (image_path, mask_path) in enumerate(items, start=1):
        name = f"{prefix}_{index:04d}.png"
        save_pair(image_path, mask_path, image_out_dir / name, mask_out_dir / name)
        names.append(name)
    return names


def clear_external_processed(dataset_name: str):
    for subdir in ["images", "masks"]:
        target = ROOT / "data" / "processed" / "test" / dataset_name / subdir
        target.mkdir(parents=True, exist_ok=True)
        for file_path in target.glob("*"):
            if file_path.is_file():
                file_path.unlink()


def print_dataset_tree(dataset_root: Path):
    print(f"Could not detect image/mask folders for {dataset_root.relative_to(ROOT)}")
    for path in [dataset_root, *list_dirs(dataset_root)[:80]]:
        try:
            files = [p for p in path.iterdir() if p.is_file()]
        except OSError:
            files = []
        print(f"  - {path.relative_to(ROOT)} ({len(files)} files)")


def export_external_dataset(dataset_name: str, prefix: str):
    raw_root = ROOT / "data" / "raw" / dataset_name
    if not raw_root.exists():
        print(f"WARNING: raw folder missing for {dataset_name}: {raw_root.relative_to(ROOT)}")
        return None

    image_dir, mask_dir = find_image_mask_dirs(raw_root)
    if not image_dir or not mask_dir or image_dir == mask_dir:
        print_dataset_tree(raw_root)
        return None

    pairs = collect_pairs(image_dir, mask_dir)
    if not pairs:
        print(f"WARNING: no matched image/mask pairs found for {dataset_name}")
        print(f"  image dir: {image_dir.relative_to(ROOT)}")
        print(f"  mask dir:  {mask_dir.relative_to(ROOT)}")
        return None

    clear_external_processed(dataset_name)
    names = []
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        name = f"{prefix}_{index:06d}.png"
        save_pair(
            image_path,
            mask_path,
            ROOT / "data" / "processed" / "test" / dataset_name / "images" / name,
            ROOT / "data" / "processed" / "test" / dataset_name / "masks" / name,
        )
        names.append(name)

    split_name = EXTERNAL_TESTSETS[dataset_name]["split"]
    write_split_file(ROOT / "data" / "splits" / split_name, names)
    print(f"{dataset_name}: {len(names)} pairs")
    print(f"  image dir: {image_dir.relative_to(ROOT)}")
    print(f"  mask dir:  {mask_dir.relative_to(ROOT)}")
    return {
        "dataset": dataset_name,
        "count": len(names),
        "image_dir": image_dir,
        "mask_dir": mask_dir,
    }


def prepare_external_only():
    print("Preparing external test sets only. Existing train/val/internal test splits are not modified.")
    results = []
    for dataset_name, info in EXTERNAL_TESTSETS.items():
        result = export_external_dataset(dataset_name, info["prefix"])
        if result is not None:
            results.append(result)
    if not results:
        print("No external datasets were prepared.")
        sys.exit(1)
    print("External test preparation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external_only", action="store_true")
    args = parser.parse_args()

    if args.external_only:
        prepare_external_only()
        return

    kvasir_image_dir, kvasir_mask_dir = find_kvasir_dirs(ROOT)
    cvc_image_dir, cvc_mask_dir = find_cvc_dirs(ROOT)

    missing = []
    if not kvasir_image_dir or not kvasir_mask_dir:
        missing.append("Kvasir-SEG images/masks")
    if not cvc_image_dir or not cvc_mask_dir:
        missing.append("CVC-ClinicDB images/masks")
    if missing:
        print(f"Missing detected folders: {', '.join(missing)}")
        print_detection_help()
        sys.exit(1)

    kvasir_pairs = collect_pairs(kvasir_image_dir, kvasir_mask_dir)
    cvc_pairs = collect_pairs(cvc_image_dir, cvc_mask_dir)
    if not kvasir_pairs or not cvc_pairs:
        print(f"Kvasir pairs found: {len(kvasir_pairs)}")
        print(f"CVC pairs found: {len(cvc_pairs)}")
        print_detection_help()
        sys.exit(1)

    print(f"Kvasir image dir: {kvasir_image_dir.relative_to(ROOT)}")
    print(f"Kvasir mask dir:  {kvasir_mask_dir.relative_to(ROOT)}")
    print(f"CVC image dir:    {cvc_image_dir.relative_to(ROOT)}")
    print(f"CVC mask dir:     {cvc_mask_dir.relative_to(ROOT)}")
    print(f"Kvasir pairs: {len(kvasir_pairs)}")
    print(f"CVC pairs: {len(cvc_pairs)}")

    clear_processed_dirs()

    kvasir_train, kvasir_test = split_dataset(kvasir_pairs, args.seed)
    cvc_train, cvc_test = split_dataset(cvc_pairs, args.seed)

    train_pool = [("kvasir", pair) for pair in kvasir_train] + [("cvc", pair) for pair in cvc_train]
    random.Random(args.seed).shuffle(train_pool)
    n_val = max(1, int(round(len(train_pool) * 0.1))) if len(train_pool) > 1 else 0
    val_pool = train_pool[:n_val]
    train_pool = train_pool[n_val:]

    train_names, val_names = [], []
    train_counters = {"kvasir": 0, "cvc": 0}
    val_counters = {"kvasir": 0, "cvc": 0}

    for source, pair in train_pool:
        train_counters[source] += 1
        name = f"{source}_{train_counters[source]:04d}.png"
        save_pair(pair[0], pair[1], ROOT / "data/processed/train/images" / name, ROOT / "data/processed/train/masks" / name)
        train_names.append(name)

    for source, pair in val_pool:
        val_counters[source] += 1
        name = f"{source}_{val_counters[source]:04d}.png"
        save_pair(pair[0], pair[1], ROOT / "data/processed/val/images" / name, ROOT / "data/processed/val/masks" / name)
        val_names.append(name)

    test_kvasir_names = export_dataset(
        kvasir_test,
        "kvasir",
        ROOT / "data/processed/test/Kvasir/images",
        ROOT / "data/processed/test/Kvasir/masks",
    )
    test_cvc_names = export_dataset(
        cvc_test,
        "cvc",
        ROOT / "data/processed/test/CVC-ClinicDB/images",
        ROOT / "data/processed/test/CVC-ClinicDB/masks",
    )

    write_split_file(ROOT / "data/splits/train.txt", train_names)
    write_split_file(ROOT / "data/splits/val.txt", val_names)
    write_split_file(ROOT / "data/splits/test_kvasir.txt", test_kvasir_names)
    write_split_file(ROOT / "data/splits/test_cvc.txt", test_cvc_names)

    print("Data preparation complete.")
    print(f"Train: {len(train_names)}")
    print(f"Val: {len(val_names)}")
    print(f"Kvasir test: {len(test_kvasir_names)}")
    print(f"CVC test: {len(test_cvc_names)}")


if __name__ == "__main__":
    main()
