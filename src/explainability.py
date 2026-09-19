"""Explainable AI (XAI) engine: Grad-CAM for CNNs and Attention Rollout for Vision Transformers."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from src.config import XAI_DIR, get_device
from src.preprocessing import get_val_test_transforms

logger = logging.getLogger("PlantCare.XAI")


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN architectures (ResNet50, EfficientNet-B0).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive influence

        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers (ViT-B/16) (Abnar & Zuidema, 2020).
    """

    def __init__(self, model: torch.nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.1):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_matrices: list[torch.Tensor] = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # Check for MultiheadAttention or self_attention layers
            if "self_attention" in name or "encoder_layer" in name:
                if hasattr(module, "register_forward_hook"):
                    hook = module.register_forward_hook(self._save_attention)
                    self.hooks.append(hook)

    def _save_attention(self, module, input, output):
        # If output is a tuple (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.attention_matrices.append(output[1].detach().cpu())

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Computes attention rollout map across transformer encoder layers.
        """
        self.model.eval()
        self.attention_matrices = []

        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attention_matrices:
            # Fallback: compute simulated patch attention using self-similarity if attention matrices not directly exposed
            return self._fallback_spatial_rollout(input_tensor)

        # Rollout computation
        result = torch.eye(self.attention_matrices[0].size(-1))
        with torch.no_grad():
            for attn in self.attention_matrices:
                if self.head_fusion == "mean":
                    attn_fused = torch.mean(attn, dim=1)
                elif self.head_fusion == "max":
                    attn_fused = torch.max(attn, dim=1)[0]
                else:
                    attn_fused = torch.mean(attn, dim=1)

                # Add identity matrix for residual connection & re-normalize
                I = torch.eye(attn_fused.size(-1))
                a = 0.5 * attn_fused[0] + 0.5 * I
                a = a / a.sum(dim=-1, keepdim=True)
                result = torch.matmul(a, result)

            mask = result[0, 1:]  # Attention from CLS token to spatial patch tokens
            num_patches = int(np.sqrt(mask.size(0)))
            mask = mask.reshape(num_patches, num_patches).numpy()

            mask_min, mask_max = mask.min(), mask.max()
            if mask_max - mask_min > 1e-8:
                mask = (mask - mask_min) / (mask_max - mask_min)
            else:
                mask = np.zeros_like(mask)

            # Resize to 224x224
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
            return np.array(mask_pil, dtype=np.float32) / 255.0

    def _fallback_spatial_rollout(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Spatial gradient-based saliency fallback for ViT architectures."""
        tensor = input_tensor.clone().detach().requires_grad_(True)
        out = self.model(tensor)
        score = torch.max(out)
        score.backward()

        saliency = torch.max(tensor.grad.data.abs(), dim=1)[0].squeeze().cpu().numpy()
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        return saliency

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Blends a 2D normalized heatmap [0, 1] with the original RGB leaf image.
    """
    rgb_img = image.convert("RGB").resize((224, 224))
    np_img = np.array(rgb_img, dtype=np.float32) / 255.0

    # Get colormap compatible with modern matplotlib versions
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = cm.get_cmap(colormap_name) if hasattr(cm, "get_cmap") else cm.jet
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    blended = (1 - alpha) * np_img + alpha * colored_heatmap
    blended = np.clip(blended, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def generate_explanation(
    model: torch.nn.Module,
    image: Image.Image,
    model_name: str,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Image.Image, np.ndarray, int, float]:
    """
    Unified entry point generating XAI explanation (Grad-CAM for CNNs, Attention Rollout for ViTs).
    Returns (overlay_image, heatmap, predicted_class_idx, confidence_score).
    """
    target_device = device or get_device()
    model.to(target_device)
    model.eval()

    transform = get_val_test_transforms(224)
    input_tensor = transform(image).unsqueeze(0).to(target_device)

    # Compute prediction & confidence
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    actual_target = target_class if target_class is not None else pred_class
    model_key = model_name.lower().replace("-", "_")

    if "vit" in model_key:
        rollout = AttentionRollout(model)
        heatmap = rollout.generate(input_tensor)
        rollout.remove_hooks()
    else:
        # Resolve target layer for CNN Grad-CAM
        if "resnet" in model_key:
            target_layer = getattr(model, "layer4")[-1]
        elif "efficientnet" in model_key:
            target_layer = model.features[-1]
        else:
            # Default to last convolutional module
            conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            target_layer = conv_layers[-1] if conv_layers else list(model.children())[-2]

        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam.generate(input_tensor, target_class=actual_target)
        grad_cam.remove_hooks()

    overlay = overlay_heatmap_on_image(image, heatmap)
    return overlay, heatmap, pred_class, confidence
