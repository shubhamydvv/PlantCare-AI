"""End-to-end integration test validating app components, UI helpers, and full diagnosis pipeline."""

from pathlib import Path
from PIL import Image
import pytest

from src.config import PLANT_VILLAGE_CLASSES, get_device, format_class_name
from src.models import get_model
from src.explainability import generate_explanation
from src.utils import validate_image_upload, compute_image_hash
from src.recommendations import (
    get_recommendation_for_disease,
    log_diagnosis_to_history,
    get_recent_diagnoses,
)
from app.components import (
    get_text,
    get_confidence_band,
    render_3d_plant_hero,
    render_3d_xai_leaf,
    render_confidence_indicator,
    render_safety_alert,
    render_history_gallery,
    TRANSLATIONS,
)


def test_theme_and_translations():
    """Validates translation dictionaries and localized text retrieval."""
    assert get_text("title", "en") != ""
    assert get_text("title", "hi") != ""
    assert get_text("step_1", "en") == "1. Upload Photo"
    assert get_text("step_1", "hi") == "1. फोटो अपलोड"
    assert get_text("disclaimer", "en").startswith("⚠️")


def test_confidence_bands_and_thresholding():
    """Tests all three confidence bands and safety thresholding."""
    high_name, high_badge, high_color = get_confidence_band(0.85, threshold=0.60)
    assert high_badge == "badge-high"
    assert high_color == "#16A34A"

    mod_name, mod_badge, mod_color = get_confidence_band(0.65, threshold=0.60)
    assert mod_badge == "badge-mod"
    assert mod_color == "#D97706"

    low_name, low_badge, low_color = get_confidence_band(0.40, threshold=0.60)
    assert low_badge == "badge-low"
    assert low_color == "#DC2626"

    # With higher threshold
    strict_name, strict_badge, strict_color = get_confidence_band(0.70, threshold=0.80)
    assert strict_badge == "badge-low"


def test_full_diagnosis_flow_with_image():
    """Tests complete image validation -> model inference -> XAI heatmap -> recommendations -> history logging."""
    test_img_path = Path("data/robustness_test_set/blur_mild/deg_00000_sample_0005.jpg")
    assert test_img_path.exists(), "Sample test image must exist"

    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    # 1. Validation
    is_valid, img, err = validate_image_upload(file_bytes, "deg_00000_sample_0005.jpg")
    assert is_valid is True
    assert isinstance(img, Image.Image)

    # 2. Inference & XAI
    device = get_device()
    model = get_model("resnet50", num_classes=len(PLANT_VILLAGE_CLASSES), pretrained=False)
    model.to(device)
    model.eval()

    overlay, heatmap, pred_idx, conf = generate_explanation(
        model=model,
        image=img,
        model_name="resnet50",
        device=device,
    )

    assert overlay is not None
    assert heatmap is not None
    assert 0 <= pred_idx < len(PLANT_VILLAGE_CLASSES)
    assert 0.0 <= conf <= 1.0

    pred_class = PLANT_VILLAGE_CLASSES[pred_idx]
    plant_name, disease_name = format_class_name(pred_class)
    assert plant_name != ""
    assert disease_name != ""

    # 3. Recommendations
    rec = get_recommendation_for_disease(pred_class)
    assert "symptoms" in rec
    assert "prevention" in rec
    assert "treatment" in rec
    assert "source_citation" in rec

    # 4. History Logging
    img_hash = compute_image_hash(img)
    row_id = log_diagnosis_to_history(
        image_hash=img_hash,
        plant_name=plant_name,
        predicted_disease=disease_name,
        confidence=conf,
        model_name="resnet50",
        flagged_uncertain=(conf < 0.60),
    )
    assert row_id > 0

    # 5. History Retrieval
    history = get_recent_diagnoses(limit=5)
    assert len(history) > 0
    assert history[0]["plant_name"] == plant_name
    assert history[0]["predicted_disease"] == disease_name
