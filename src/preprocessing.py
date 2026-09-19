"""Preprocessing, data augmentation, and stratified dataset splitting."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from torch.utils.data import DataLoader

from src.config import (
    CONFIG,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLANT_VILLAGE_CLASSES,
    set_seed,
)
from src.dataset import PlantVillageDataset, generate_mock_dataset

logger = logging.getLogger(__name__)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Returns training transformation pipeline with augmentations."""
    aug_cfg = CONFIG.get("data", {}).get("augmentation", {})
    norm_cfg = CONFIG.get("data", {}).get("normalization", {})

    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]

    if aug_cfg.get("horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip", True):
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    if aug_cfg.get("rotation_degrees", 0) > 0:
        transform_list.append(transforms.RandomRotation(degrees=aug_cfg["rotation_degrees"]))
    if "color_jitter" in aug_cfg:
        cj = aug_cfg["color_jitter"]
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.1),
                contrast=cj.get("contrast", 0.1),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.05),
            )
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)


def get_val_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Returns deterministic validation / test transformation pipeline."""
    norm_cfg = CONFIG.get("data", {}).get("normalization", {})
    mean = norm_cfg.get("mean", [0.485, 0.456, 0.406])
    std = norm_cfg.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def scan_dataset(raw_dir: Path | str = RAW_DATA_DIR) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans raw dataset directory and returns (image_paths, label_indices, class_names).
    """
    raw_path = Path(raw_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Discover classes present on disk
    existing_classes = sorted([
        d.name for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ])

    if not existing_classes:
        logger.warning(f"No class folders found in {raw_path}. Generating synthetic mock dataset...")
        generate_mock_dataset(raw_path, num_samples_per_class=15)
        existing_classes = sorted([
            d.name for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ])

    class_to_idx = {cls: idx for idx, cls in enumerate(existing_classes)}

    image_paths: List[str] = []
    labels: List[int] = []

    for cls_name in existing_classes:
        cls_dir = raw_path / cls_name
        cls_idx = class_to_idx[cls_name]
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in image_extensions:
                image_paths.append(str(f.resolve()))
                labels.append(cls_idx)

    return image_paths, labels, existing_classes


def create_stratified_splits(
    image_paths: List[str],
    labels: List[int],
    class_names: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Performs 70/15/15 stratified splitting and optionally persists split metadata.
    """
    set_seed(seed)
    np_labels = np.array(labels)
    indices = np.arange(len(image_paths))

    # Split 1: Split into train (70%) and temporary val+test (30%)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, np_labels))

    # Split 2: Split temporary (30%) equally into val (15%) and test (15%)
    val_share = val_ratio / temp_ratio
    temp_labels = np_labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_share), random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_idx, temp_labels))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    split_data = {
        "seed": seed,
        "class_names": class_names,
        "num_classes": len(class_names),
        "total_images": len(image_paths),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train": {
            "image_paths": [image_paths[i] for i in train_idx],
            "labels": [int(labels[i]) for i in train_idx],
        },
        "val": {
            "image_paths": [image_paths[i] for i in val_idx],
            "labels": [int(labels[i]) for i in val_idx],
        },
        "test": {
            "image_paths": [image_paths[i] for i in test_idx],
            "labels": [int(labels[i]) for i in test_idx],
        },
    }

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)
        logger.info(f"Saved stratified split info to {out_file}")

    return split_data


def load_or_create_splits(
    split_info_path: Path | str = PROCESSED_DATA_DIR / "split_info.json",
    raw_dir: Path | str = RAW_DATA_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads existing persisted split info or generates new stratified splits."""
    split_file = Path(split_info_path)
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            return json.load(f)

    paths, labels, class_names = scan_dataset(raw_dir)
    return create_stratified_splits(
        paths,
        labels,
        class_names,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        save_path=split_file,
    )


def get_dataloaders(
    split_data: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds PyTorch DataLoaders for train, val, and test splits."""
    train_tf = get_train_transforms(image_size)
    val_tf = get_val_test_transforms(image_size)

    train_ds = PlantVillageDataset(
        split_data["train"]["image_paths"],
        split_data["train"]["labels"],
        transform=train_tf,
    )
    val_ds = PlantVillageDataset(
        split_data["val"]["image_paths"],
        split_data["val"]["labels"],
        transform=val_tf,
    )
    test_ds = PlantVillageDataset(
        split_data["test"]["image_paths"],
        split_data["test"]["labels"],
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    splits = load_or_create_splits()
    print(f"Data pipeline initialized: {splits['train_size']} train, {splits['val_size']} val, {splits['test_size']} test images across {splits['num_classes']} classes.")
