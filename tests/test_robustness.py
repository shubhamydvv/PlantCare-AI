"""Unit tests for the 4 robustness degradation transformations."""

import pytest
import numpy as np
from PIL import Image

from src.robustness import (
    apply_brightness,
    apply_gaussian_blur,
    apply_rotation,
    apply_gaussian_noise,
    ROBUSTNESS_TRANSFORMS,
)


@pytest.fixture
def sample_image():
    """Returns a synthetic test RGB image."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    return img


def test_brightness_transform(sample_image):
    """Tests brightness shift low and high."""
    low = apply_brightness(sample_image, 0.60)
    high = apply_brightness(sample_image, 1.40)

    assert low.size == (224, 224)
    assert high.size == (224, 224)
    # Low brightness should have lower mean pixel intensity than high
    assert np.mean(np.array(low)) < np.mean(np.array(high))


def test_gaussian_blur_transform(sample_image):
    """Tests Gaussian blur."""
    blurred = apply_gaussian_blur(sample_image, radius=2.0)
    assert blurred.size == (224, 224)


def test_rotation_transform(sample_image):
    """Tests rotation transform."""
    for angle in [45, 90, 180]:
        rotated = apply_rotation(sample_image, angle)
        assert rotated.size == (224, 224)


def test_gaussian_noise_transform(sample_image):
    """Tests noise addition."""
    noisy = apply_gaussian_noise(sample_image, std=0.10, seed=42)
    assert noisy.size == (224, 224)
    # Noisy image array should differ from original
    assert not np.array_equal(np.array(sample_image), np.array(noisy))


def test_all_registry_transforms_execute(sample_image):
    """Verifies that every transform in the 9-item registry executes cleanly."""
    assert len(ROBUSTNESS_TRANSFORMS) == 9
    for name, transform_fn in ROBUSTNESS_TRANSFORMS.items():
        out = transform_fn(sample_image)
        assert isinstance(out, Image.Image)
        assert out.size == (224, 224)
