"""Model architecture definitions: ResNet50, EfficientNet-B0, ViT-B/16, and CNN-ViT Hybrid."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = ["resnet50", "efficientnet_b0", "vit_b_16", "cnn_vit_hybrid"]


class CNNViTHybrid(nn.Module):
    """
    CNN-ViT Hybrid Model combining ResNet feature extraction with a Transformer encoder
    (inspired by Borhani et al. 2022 & Vision Transformer Meets CNN).
    """

    def __init__(self, num_classes: int, pretrained: bool = True, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        # Use initial layers up to layer3: produces (B, 256, 14, 14) for 224x224 input
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14 + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feats = self.feature_extractor(x)  # (B, 256, 14, 14)
        proj = self.proj(feats)  # (B, embed_dim, 14, 14)
        tokens = proj.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_tokens = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed
        out_tokens = self.transformer(x_tokens)
        cls_out = self.norm(out_tokens[:, 0])
        return self.classifier(cls_out)


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Model factory instantiating ResNet50, EfficientNet-B0, ViT-B/16, or CNN-ViT Hybrid.
    """
    model_key = model_name.lower().replace("-", "_")

    if model_key == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logger.info(f"Initialized ResNet50 (in_features={in_features}, num_classes={num_classes}, pretrained={pretrained})")
        return model

    elif model_key in ["efficientnet_b0", "efficientnet"]:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        logger.info(f"Initialized EfficientNet-B0 (in_features={in_features}, num_classes={num_classes}, pretrained={pretrained})")
        return model

    elif model_key in ["vit_b_16", "vit", "vit_b16"]:
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        logger.info(f"Initialized ViT-B/16 (in_features={in_features}, num_classes={num_classes}, pretrained={pretrained})")
        return model

    elif model_key in ["cnn_vit_hybrid", "hybrid"]:
        model = CNNViTHybrid(num_classes=num_classes, pretrained=pretrained)
        logger.info(f"Initialized CNN-ViT Hybrid (num_classes={num_classes}, pretrained={pretrained})")
        return model

    else:
        raise ValueError(
            f"Unsupported model architecture '{model_name}'. Choose from: {SUPPORTED_ARCHITECTURES}"
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns total, trainable, and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }


def load_model_checkpoint(
    model_name: str,
    num_classes: int,
    checkpoint_path: Path | str,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Loads model weights from a persisted checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    target_device = device or torch.device("cpu")
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(ckpt_file, map_location=target_device)

    # Unwrap state dict if saved under 'model_state_dict' or raw
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to(target_device)
    model.eval()
    logger.info(f"Loaded checkpoint for {model_name} from {ckpt_file}")
    return model
