"""Script to generate sample XAI overlay outputs across all models."""

import json
from pathlib import Path
from PIL import Image
import torch

from src.config import XAI_DIR, PLANT_VILLAGE_CLASSES, get_device
from src.models import load_model_checkpoint
from src.explainability import generate_explanation
from src.preprocessing import load_or_create_splits

def generate_sample_xai_outputs():
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    splits = load_or_create_splits()
    test_paths = splits["test"]["image_paths"]
    num_classes = len(PLANT_VILLAGE_CLASSES)

    # Pick 4 representative test images
    sample_images = [test_paths[i] for i in range(min(4, len(test_paths)))]
    models = ["resnet50", "efficientnet_b0", "vit_b_16"]

    for model_name in models:
        ckpt = Path("models") / model_name / "best_model.pth"
        model = load_model_checkpoint(model_name, num_classes=num_classes, checkpoint_path=ckpt, device=device)

        for idx, img_p in enumerate(sample_images):
            img = Image.open(img_p).convert("RGB")
            overlay, heatmap, pred_idx, conf = generate_explanation(model, img, model_name=model_name, device=device)
            save_name = XAI_DIR / f"{model_name}_sample_{idx:02d}_{Path(img_p).stem}_overlay.jpg"
            overlay.save(save_name, "JPEG", quality=95)
            print(f"Saved XAI overlay: {save_name} (Pred: {PLANT_VILLAGE_CLASSES[pred_idx]}, Conf: {conf:.2%})")

if __name__ == "__main__":
    generate_sample_xai_outputs()
