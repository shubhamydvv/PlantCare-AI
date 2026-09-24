"""Script to test end-to-end plant disease diagnosis and XAI on leaf image."""

import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from src.config import (
    PLANT_VILLAGE_CLASSES,
    get_device,
    format_class_name,
    XAI_DIR,
    DB_PATH,
)
from src.models import get_model, load_model_checkpoint
from src.explainability import generate_explanation
from src.utils import validate_image_upload, compute_image_hash
from src.recommendations import (
    get_recommendation_for_disease,
    log_diagnosis_to_history,
    get_recent_diagnoses,
)

def run_image_test(image_path_str: str, model_name: str = "resnet50"):
    img_path = Path(image_path_str)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return

    print("=" * 70)
    print("🌿 PLANTCARE AI - DIAGNOSTIC PIPELINE TEST")
    print("=" * 70)
    print(f"Input Leaf Specimen: {img_path.name}")
    print(f"File Path:          {img_path.resolve()}")
    print(f"Selected AI Model:  {model_name}")

    # 1. Validation & Hashing
    with open(img_path, "rb") as f:
        raw_bytes = f.read()

    is_valid, pil_img, err_msg = validate_image_upload(raw_bytes, img_path.name)
    if not is_valid:
        print(f"❌ Image Validation Failed: {err_msg}")
        return

    img_hash = compute_image_hash(pil_img)
    print(f"Image Resolution:   {pil_img.size[0]} x {pil_img.size[1]} pixels (RGB)")
    print(f"Specimen Hash:      {img_hash}")

    # 2. Model Loading & Inference
    device = get_device()
    print(f"Compute Device:     {device.type.upper()}")
    num_classes = len(PLANT_VILLAGE_CLASSES)
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)
    model.eval()

    start_t = time.time()
    overlay_img, heatmap_img, pred_idx, confidence = generate_explanation(
        model=model,
        image=pil_img,
        model_name=model_name,
        device=device,
    )
    latency_ms = (time.time() - start_t) * 1000

    raw_class = PLANT_VILLAGE_CLASSES[pred_idx] if pred_idx < len(PLANT_VILLAGE_CLASSES) else "Unknown"
    plant_name, disease_name = format_class_name(raw_class)

    # 3. Save XAI Output
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    overlay_out_path = XAI_DIR / f"test_output_{img_path.stem}_{model_name}_xai.jpg"
    overlay_img.save(overlay_out_path)

    # 4. Knowledge Base Query
    rec = get_recommendation_for_disease(raw_class, db_path=DB_PATH)

    # 5. SQLite Diagnosis History Log
    row_id = log_diagnosis_to_history(
        image_hash=img_hash,
        plant_name=plant_name,
        predicted_disease=disease_name,
        confidence=confidence,
        model_name=model_name,
        flagged_uncertain=(confidence < 0.60),
        notes="Automated CLI diagnostic test",
        db_path=DB_PATH,
    )

    # 6. Output Diagnostic Results
    print("\n" + "-" * 70)
    print("🔬 DIAGNOSTIC INFERENCE & EXPLAINABLE AI RESULT")
    print("-" * 70)
    print(f"• Predicted Class:       {raw_class}")
    print(f"• Target Crop Species:   {plant_name}")
    print(f"• Identified Pathology:  {disease_name}")
    print(f"• Confidence Score:      {confidence * 100:.2f}%")
    print(f"• Safety Status:         {'⚠️ UNCERTAIN (<60%)' if confidence < 0.60 else '✅ HIGH CONFIDENCE'}")
    print(f"• Inference Latency:     {latency_ms:.2f} ms")
    print(f"• Grad-CAM Overlay:      {overlay_out_path.name} (Saved to results/xai_outputs/)")
    print(f"• Logged to History ID:  #{row_id}")

    print("\n" + "-" * 70)
    print("📚 EVIDENCE-BASED AGRONOMIC GUIDANCE (SQLite Knowledge Base)")
    print("-" * 70)
    print(f"🔍 SYMPTOMS:\n   {rec.get('symptoms', 'N/A')}")
    print(f"\n🛡️ CULTURAL & PREVENTATIVE PROTOCOL:\n   {rec.get('prevention', 'N/A')}")
    print(f"\n💊 THERAPEUTIC TREATMENT GUIDANCE:\n   {rec.get('treatment', 'N/A')}")
    print(f"\n📖 EXTENSION CITATION:\n   {rec.get('source_citation', 'N/A')}")
    print("=" * 70)

if __name__ == "__main__":
    sample_img = "data/robustness_test_set/blur_mild/deg_00000_sample_0005.jpg"
    if len(sys.argv) > 1:
        sample_img = sys.argv[1]
    arch = "resnet50"
    if len(sys.argv) > 2:
        arch = sys.argv[2]
    run_image_test(sample_img, arch)
