"""Unit tests for dataset preprocessing, augmentation, and stratified splitting."""

import pytest
import numpy as np
import torch
from PIL import Image

from src.config import PLANT_VILLAGE_CLASSES
from src.preprocessing import (
    get_train_transforms,
    get_val_test_transforms,
    create_stratified_splits,
)
from src.dataset import PlantVillageDataset


def test_train_transforms_output():
    """Verifies that train transform pipeline outputs normalized (3, 224, 224) float tensor."""
    tf = get_train_transforms(224)
    img = Image.new("RGB", (300, 300), color=(100, 150, 50))
    tensor = tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_val_test_transforms_output():
    """Verifies that validation/test transform pipeline outputs (3, 224, 224) tensor."""
    tf = get_val_test_transforms(224)
    img = Image.new("RGB", (400, 300), color=(50, 100, 150))
    tensor = tf(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_stratified_split_proportions():
    """Verifies that stratified splitting preserves 70/15/15 distribution and class balance."""
    num_classes = 5
    samples_per_class = 20
    paths = []
    labels = []

    for c in range(num_classes):
        for s in range(samples_per_class):
            paths.append(f"fake/path/class_{c}/img_{s}.jpg")
            labels.append(c)

    class_names = [f"Class_{c}" for c in range(num_classes)]
    splits = create_stratified_splits(
        paths,
        labels,
        class_names,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    total = len(paths)
    assert splits["train_size"] == 70
    assert splits["val_size"] == 15
    assert splits["test_size"] == 15
    assert splits["train_size"] + splits["val_size"] + splits["test_size"] == total

    # Check that each class is represented in train, val, and test
    for c in range(num_classes):
        assert splits["train"]["labels"].count(c) == 14
        assert splits["val"]["labels"].count(c) == 3
        assert splits["test"]["labels"].count(c) == 3


def test_dataset_item_retrieval(tmp_path):
    """Verifies that PlantVillageDataset returns valid (image_tensor, label) pairs."""
    img_file = tmp_path / "test_leaf.jpg"
    Image.new("RGB", (100, 100), color=(30, 180, 40)).save(img_file)

    tf = get_val_test_transforms(224)
    ds = PlantVillageDataset([str(img_file)], [2], transform=tf)

    assert len(ds) == 1
    img_t, label_t = ds[0]
    assert img_t.shape == (3, 224, 224)
    assert label_t.item() == 2
