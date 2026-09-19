"""Dataset loading, programmatic fetching, synthetic mock generation, and PyTorch Dataset."""

from __future__ import annotations
import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLANT_VILLAGE_CLASSES,
    set_seed,
)

logger = logging.getLogger(__name__)


class PlantVillageDataset(Dataset):
    """PyTorch Dataset for PlantCare AI leaf images."""

    def __init__(
        self,
        image_paths: List[str | Path],
        labels: List[int],
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
        self.return_path = return_path

        if len(self.image_paths) != len(self.labels):
            raise ValueError("Number of images and labels must match.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Corrupt or unreadable image at {img_path}: {e}")
            # Fallback to a blank image if file is corrupt
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.return_path:
            return image, label, str(img_path)
        return image, label


def generate_mock_dataset(
    target_dir: Path | str = RAW_DATA_DIR,
    num_samples_per_class: int = 15,
    classes: Optional[List[str]] = None,
    seed: int = 42,
) -> Path:
    """
    Generates a realistic synthetic mock PlantVillage leaf image dataset for local testing,
    continuous integration, and rapid dry-run experimentation.
    """
    set_seed(seed)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    class_list = classes or PLANT_VILLAGE_CLASSES

    np.random.seed(seed)
    for class_name in class_list:
        class_folder = target_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        is_healthy = "healthy" in class_name.lower()

        for i in range(num_samples_per_class):
            img_file = class_folder / f"sample_{i:04d}.jpg"
            if img_file.exists():
                continue

            # Create synthetic leaf canvas (256x256)
            img = Image.new("RGB", (256, 256), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Base leaf green color with random variation
            g_val = np.random.randint(120, 190)
            r_val = np.random.randint(30, 80)
            b_val = np.random.randint(20, 60)
            leaf_color = (r_val, g_val, b_val)

            # Draw leaf elliptical shape
            leaf_bbox = [
                np.random.randint(25, 45),
                np.random.randint(25, 45),
                np.random.randint(210, 235),
                np.random.randint(210, 235),
            ]
            draw.ellipse(leaf_bbox, fill=leaf_color, outline=(20, 90, 20), width=2)

            # Draw primary and secondary leaf veins
            vein_color = (max(0, r_val - 20), min(255, g_val + 30), max(0, b_val - 10))
            draw.line([(128, 40), (128, 220)], fill=vein_color, width=2)
            for y_vein in range(60, 210, 25):
                draw.line([(128, y_vein), (70, y_vein - 20)], fill=vein_color, width=1)
                draw.line([(128, y_vein), (186, y_vein - 20)], fill=vein_color, width=1)

            # If diseased, generate disease lesions/spots
            if not is_healthy:
                num_spots = np.random.randint(4, 12)
                for _ in range(num_spots):
                    spot_x = np.random.randint(60, 190)
                    spot_y = np.random.randint(60, 190)
                    spot_r = np.random.randint(6, 20)

                    # Brown/yellow/black lesion colors
                    if "rust" in class_name.lower():
                        spot_color = (np.random.randint(180, 220), np.random.randint(80, 120), 20)
                    elif "blight" in class_name.lower() or "rot" in class_name.lower():
                        spot_color = (np.random.randint(50, 90), np.random.randint(30, 60), 20)
                    elif "mildew" in class_name.lower() or "mold" in class_name.lower():
                        spot_color = (220, 220, 200)
                    else:
                        spot_color = (np.random.randint(120, 160), np.random.randint(80, 110), 30)

                    draw.ellipse(
                        [spot_x - spot_r, spot_y - spot_r, spot_x + spot_r, spot_y + spot_r],
                        fill=spot_color,
                    )

            # Apply light smoothing filter
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img.save(img_file, "JPEG", quality=90)

    logger.info(f"Generated synthetic mock dataset at {target_path} across {len(class_list)} classes.")
    return target_path


def download_plantvillage_kaggle(target_dir: Path | str = RAW_DATA_DIR) -> Optional[Path]:
    """
    Downloads PlantVillage dataset from Kaggle via kagglehub if installed and configured.
    Falls back gracefully if network or kaggle credentials are not active.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        logger.info("Attempting automated dataset download via kagglehub...")
        path = kagglehub.dataset_download("emmarex/plantdisease")
        logger.info(f"Downloaded raw dataset from kagglehub to cache: {path}")

        # Scan for class subdirectories and symlink or copy
        downloaded_path = Path(path)
        return downloaded_path
    except Exception as e:
        logger.warning(f"KaggleHub automated download skipped/failed ({e}). Manual download instructions in README.")
        return None
