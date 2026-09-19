"""Robustness degradation transforms, fixed dataset generation, and robustness loader."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import (
    CONFIG,
    ROBUSTNESS_DATA_DIR,
    PROCESSED_DATA_DIR,
    set_seed,
)
from src.preprocessing import load_or_create_splits

logger = logging.getLogger(__name__)


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Applies brightness scaling factor (e.g. 0.60 for low, 1.40 for high)."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Applies Gaussian blur with given radius."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_rotation(image: Image.Image, angle: float) -> Image.Image:
    """Applies rotation around center with bilinear resampling."""
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)


def apply_gaussian_noise(image: Image.Image, std: float, seed: int = 42) -> Image.Image:
    """Adds zero-mean Gaussian noise with given standard deviation (scaled in [0, 1])."""
    np_img = np.array(image, dtype=np.float32) / 255.0
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=std, size=np_img.shape)
    noisy_img = np.clip(np_img + noise, 0.0, 1.0)
    return Image.fromarray((noisy_img * 255.0).astype(np.uint8))


# Registry of the 9 specific degradation variants
ROBUSTNESS_TRANSFORMS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "brightness_low": lambda img: apply_brightness(img, 0.60),
    "brightness_high": lambda img: apply_brightness(img, 1.40),
    "blur_mild": lambda img: apply_gaussian_blur(img, 1.5),
    "blur_heavy": lambda img: apply_gaussian_blur(img, 3.0),
    "rotation_45": lambda img: apply_rotation(img, 45),
    "rotation_90": lambda img: apply_rotation(img, 90),
    "rotation_180": lambda img: apply_rotation(img, 180),
    "noise_mild": lambda img: apply_gaussian_noise(img, 0.05),
    "noise_heavy": lambda img: apply_gaussian_noise(img, 0.15),
}


def generate_robustness_test_sets(
    split_info: Dict[str, Any],
    output_dir: Path | str = ROBUSTNESS_DATA_DIR,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generates and persists the 4 degradation variants across all severity levels
    from the fixed clean test split.
    """
    set_seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_paths = split_info["test"]["image_paths"]
    test_labels = split_info["test"]["labels"]
    class_names = split_info["class_names"]

    manifest: Dict[str, Any] = {
        "seed": seed,
        "class_names": class_names,
        "num_classes": len(class_names),
        "total_clean_test_images": len(test_paths),
        "variants": {},
    }

    for variant_name, transform_fn in ROBUSTNESS_TRANSFORMS.items():
        variant_dir = out_path / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_image_paths = []

        for idx, (img_path_str, label) in enumerate(zip(test_paths, test_labels)):
            orig_path = Path(img_path_str)
            target_filename = f"deg_{idx:05d}_{orig_path.stem}.jpg"
            target_file = variant_dir / target_filename

            if not target_file.exists():
                try:
                    img = Image.open(orig_path).convert("RGB")
                    deg_img = transform_fn(img)
                    deg_img.save(target_file, "JPEG", quality=90)
                except Exception as e:
                    logger.error(f"Error degrading {orig_path} for {variant_name}: {e}")
                    continue

            variant_image_paths.append(str(target_file.resolve()))

        manifest["variants"][variant_name] = {
            "image_paths": variant_image_paths,
            "labels": test_labels[:len(variant_image_paths)],
            "count": len(variant_image_paths),
        }
        logger.info(f"Generated {len(variant_image_paths)} images for variant: {variant_name}")

    manifest_path = out_path / "robustness_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved complete robustness manifest to {manifest_path}")
    return manifest


def load_or_generate_robustness_manifest(
    output_dir: Path | str = ROBUSTNESS_DATA_DIR,
    split_info_path: Path | str = PROCESSED_DATA_DIR / "split_info.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads existing robustness manifest or generates it from test split."""
    manifest_path = Path(output_dir) / "robustness_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    split_info = load_or_create_splits(split_info_path=split_info_path, seed=seed)
    return generate_robustness_test_sets(split_info, output_dir=output_dir, seed=seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manifest = load_or_generate_robustness_manifest()
    print(f"Robustness benchmark generated across {len(manifest['variants'])} degradation variants.")
