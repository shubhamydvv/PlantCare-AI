"""Utility helpers: image validation, hashing, ONNX export, and logging."""

from __future__ import annotations
import hashlib
import io
import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import torch

logger = logging.getLogger("PlantCare.Utils")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


def compute_image_hash(image: Image.Image | bytes) -> str:
    """Computes a deterministic SHA-256 hash of an image for deduplication and logging."""
    if isinstance(image, bytes):
        return hashlib.sha256(image).hexdigest()[:16]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def validate_image_upload(
    file_bytes: bytes,
    filename: str = "upload.jpg",
) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Validates uploaded image file bytes: checks file size, extension, and corrupt data.
    Returns (is_valid, PIL_Image_or_None, error_message).
    """
    # 1. Size check
    if len(file_bytes) == 0:
        return False, None, "Uploaded file is empty (0 bytes)."
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, None, f"File size ({len(file_bytes) / 1e6:.1f} MB) exceeds 10MB limit."

    # 2. Extension check
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False, None, f"Unsupported image extension '{ext}'. Please upload JPG, PNG, or WebP."

    # 3. Integrity check
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # Verify integrity
        # Re-open after verify() per PIL specification
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return True, img, ""
    except Exception as e:
        return False, None, f"Corrupted or unreadable image file: {e}"


def export_model_to_onnx(
    model: torch.nn.Module,
    save_path: Path | str,
    image_size: int = 224,
    device: Optional[torch.device] = None,
) -> Path:
    """
    Exports a trained PyTorch model to ONNX format for high-speed edge and CPU deployment.
    """
    target_device = device or torch.device("cpu")
    model.to(target_device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=target_device)
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["class_logits"],
            dynamic_axes={"input_image": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        )
        logger.info(f"Successfully exported ONNX model to {out_file}")
        return out_file
    except Exception as e:
        logger.warning(f"ONNX export requires onnx and onnxscript packages ({e}).")
        return out_file
