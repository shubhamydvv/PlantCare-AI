"""Configuration loader and system path constants for PlantCare AI."""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any, Dict, List
import yaml
import numpy as np
import torch

# Base project directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ROBUSTNESS_DATA_DIR = DATA_DIR / "robustness_test_set"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
GRAPHS_DIR = RESULTS_DIR / "graphs"
XAI_DIR = RESULTS_DIR / "xai_outputs"
DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "plantcare.db"

# Create directories if they do not exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ROBUSTNESS_DATA_DIR,
    MODELS_DIR / "resnet50",
    MODELS_DIR / "efficientnet",
    MODELS_DIR / "vit",
    METRICS_DIR,
    GRAPHS_DIR,
    XAI_DIR,
    DATABASE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_yaml_config(config_path: Path | str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Global config instance
CONFIG = load_yaml_config()


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Canonical 38 PlantVillage disease classes with readable display names
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def format_class_name(raw_name: str) -> tuple[str, str]:
    """
    Parses raw class name into (Plant Species, Disease / Healthy).
    Example: 'Tomato___Early_blight' -> ('Tomato', 'Early Blight')
    """
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ").replace(",", "").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, disease
