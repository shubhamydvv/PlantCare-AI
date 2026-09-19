"""Unit tests for model architectures and parameter shapes."""

import pytest
import torch

from src.models import get_model, count_parameters, SUPPORTED_ARCHITECTURES


@pytest.mark.parametrize("model_name", ["resnet50", "efficientnet_b0", "vit_b_16", "cnn_vit_hybrid"])
def test_model_instantiation_and_forward(model_name):
    """Verifies that all architectures output logits of shape (B, num_classes)."""
    num_classes = 38
    batch_size = 2
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, num_classes)


def test_count_parameters():
    """Verifies that parameter count helper returns valid non-zero counts."""
    model = get_model("resnet50", num_classes=10, pretrained=False)
    counts = count_parameters(model)

    assert "total_params" in counts
    assert "trainable_params" in counts
    assert counts["total_params"] > 1_000_000


def test_invalid_architecture_name():
    """Verifies that unsupported architecture names raise ValueError."""
    with pytest.raises(ValueError):
        get_model("unknown_arch_xyz", num_classes=10)
