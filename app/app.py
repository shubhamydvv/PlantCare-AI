"""PlantCare AI - Interactive Streamlit Web Application with Explainable AI & 3D Visualization."""

from __future__ import annotations
import io
import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONFIG,
    MODELS_DIR,
    METRICS_DIR,
    GRAPHS_DIR,
    DB_PATH,
    PLANT_VILLAGE_CLASSES,
    get_device,
    format_class_name,
)
from src.models import get_model, load_model_checkpoint
from src.explainability import generate_explanation
from src.utils import validate_image_upload, compute_image_hash
from src.recommendations import (
    get_recommendation_for_disease,
    log_diagnosis_to_history,
    get_recent_diagnoses,
)
try:
    from components import (
        get_text,
        inject_custom_styles,
        render_3d_plant_hero,
        render_3d_xai_leaf,
        render_confidence_indicator,
        render_safety_alert,
        render_history_gallery,
    )
except ModuleNotFoundError:
    from app.components import (
        get_text,
        inject_custom_styles,
        render_3d_plant_hero,
        render_3d_xai_leaf,
        render_confidence_indicator,
        render_safety_alert,
        render_history_gallery,
    )

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="PlantCare AI — Explainable Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Responsive CSS & Custom Design System
inject_custom_styles()


@st.cache_resource(show_spinner="Loading AI model weights into memory...")
def load_cached_model(model_name: str):
    """Loads and caches model architecture and weights for fast inference."""
    device = get_device()
    ckpt_path = MODELS_DIR / model_name / "best_model.pth"
    num_classes = len(PLANT_VILLAGE_CLASSES)

    if ckpt_path.exists():
        model = load_model_checkpoint(model_name, num_classes=num_classes, checkpoint_path=ckpt_path, device=device)
    else:
        # Load initialized pretrained baseline
        model = get_model(model_name, num_classes=num_classes, pretrained=True)
        model.to(device)
        model.eval()

    return model, device


# Preset curated sample leaf specimens for instant 1-click testing
SAMPLE_SPECIMENS = [
    {
        "label": "🍅 Tomato Leaf (Specimen A)",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00000_sample_0005.jpg",
        "species": "Tomato",
    },
    {
        "label": "🍎 Apple Leaf (Specimen B)",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00001_sample_0008.jpg",
        "species": "Apple",
    },
    {
        "label": "🌽 Corn Foliage (Specimen C)",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00002_sample_0004.jpg",
        "species": "Corn (maize)",
    },
    {
        "label": "🍇 Grape Leaf (Specimen D)",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00003_sample_0010.jpg",
        "species": "Grape",
    },
    {
        "label": "🌿 Pepper Foliage (Specimen E)",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00004_sample_0003.jpg",
        "species": "Pepper, bell",
    },
]


def main():
    # --------------------------------------------------------------------------
    # SIDEBAR: Configuration, Architecture & Uncertainty Threshold
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #E2E8F0;">
                <span style="font-size: 2.2rem; background: #DCFCE7; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid #86EFAC;">🌿</span>
                <div>
                    <div style="font-weight: 800; font-size: 1.3rem; color: #064E3B; line-height: 1.1;">PlantCare AI</div>
                    <div style="font-size: 0.76rem; color: #16A34A; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Clinical XAI Assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lang_choice = st.selectbox(
            "🌐 Language / भाषा",
            ["English (en)", "हिन्दी (hi)"],
            index=0,
        )
        lang = "hi" if "हिन्दी" in lang_choice else "en"

        st.markdown(f"### ⚙️ {get_text('sidebar_title', lang)}")

        model_choice = st.selectbox(
            get_text("select_model", lang),
            ["resnet50", "efficientnet_b0", "vit_b_16"],
            format_func=lambda x: {
                "resnet50": "ResNet50 (Deep CNN + Grad-CAM)",
                "efficientnet_b0": "EfficientNet-B0 (Lightweight CNN)",
                "vit_b_16": "ViT-B/16 (Vision Transformer + Rollout)",
            }.get(x, x),
        )

        threshold_val = st.slider(
            get_text("confidence_threshold", lang),
            min_value=30,
            max_value=90,
            value=int(CONFIG.get("app", {}).get("confidence_threshold", 0.60) * 100),
            step=5,
            help=get_text("confidence_threshold_help", lang),
        )
        conf_threshold = threshold_val / 100.0

        st.markdown("---")
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; font-size: 0.82rem; color: #4B6354; line-height: 1.45;">
                <strong style="color: #064E3B;">🎓 Academic Research Prototype</strong><br>
                Department of Computer Science & Engineering<br>
                <em>Chandigarh University</em><br>
                <div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #E2E8F0; font-size: 0.78rem;">
                    • 38 PlantVillage Pathologies<br>
                    • PyTorch & Grad-CAM XAI<br>
                    • SQLite Knowledge Base
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # HERO / LANDING SECTION (With Embedded 3D Procedural Plant)
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="plant-hero-container">
            <div class="plant-hero-badge">🌿 {get_text('app_badge', lang)}</div>
            <div class="plant-hero-title">{get_text('hero_value_prop', lang)}</div>
            <div class="plant-hero-desc">{get_text('subtitle', lang)}</div>
            <div class="workflow-steps-container">
                <div class="workflow-step">
                    <div class="workflow-step-icon">📤</div>
                    <span>{get_text('step_1', lang)}</span>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-icon">🧠</div>
                    <span>{get_text('step_2', lang)}</span>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-icon">🔬</div>
                    <span>{get_text('step_3', lang)}</span>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-icon">📋</div>
                    <span>{get_text('step_4', lang)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Embed 3D Three.js Procedural Botanical Hero Accent
    render_3d_plant_hero(height=320)

    # Persistent Research Disclaimer
    st.warning(get_text("disclaimer", lang))

    # --------------------------------------------------------------------------
    # MAIN TABS: Diagnosis & XAI | Research Metrics | Diagnosis History
    # --------------------------------------------------------------------------
    tab_diag, tab_metrics, tab_history = st.tabs([
        "🔬 Diagnosis & XAI",
        "📊 Research Metrics & Robustness",
        "📜 Diagnosis History",
    ])

    # ==========================================================================
    # TAB 1: DIAGNOSIS & EXPLAINABLE AI
    # ==========================================================================
    with tab_diag:
        # Quick Sample Leaf Selection Tray (One-Click Testing)
        st.markdown(
            f"""
            <div class="sample-tray-container">
                <div class="sample-tray-title">⚡ {get_text('sample_tray_title', lang)}</div>
                <div style="font-size: 0.84rem; color: #4B6354; margin-bottom: 10px;">{get_text('sample_tray_hint', lang)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        cols = [col_s1, col_s2, col_s3, col_s4, col_s5]

        for i, spec in enumerate(SAMPLE_SPECIMENS):
            with cols[i]:
                if st.button(spec["label"], key=f"quick_sample_{i}", use_container_width=True):
                    if spec["file"].exists():
                        with open(spec["file"], "rb") as f:
                            st.session_state["loaded_sample_bytes"] = f.read()
                            st.session_state["loaded_sample_name"] = spec["file"].name
                            st.session_state["selected_sample_species"] = spec["species"]

        col_input, col_view = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown(f"### 📤 {get_text('select_plant', lang)}")

            # Species selector
            known_plants = sorted(list(set([c.split("___")[0].replace("_", " ").strip() for c in PLANT_VILLAGE_CLASSES])))
            default_species_idx = 0
            if "selected_sample_species" in st.session_state and st.session_state["selected_sample_species"] in known_plants:
                default_species_idx = known_plants.index(st.session_state["selected_sample_species"]) + 1

            selected_plant = st.selectbox(
                f"🌱 {get_text('select_plant', lang)}",
                ["Any / Auto-Detect"] + known_plants,
                index=default_species_idx,
                key="plant_species_select",
            )

            # Upload options with clear drag-and-drop hint
            uploaded_file = st.file_uploader(
                f"{get_text('upload_label', lang)} ({get_text('upload_drag_hint', lang)})",
                type=["jpg", "jpeg", "png", "webp"],
                help=get_text("upload_help", lang),
            )
            camera_file = st.camera_input(get_text("camera_label", lang))

            # Determine active image source (Uploaded file, Camera, or Preloaded Sample)
            active_bytes = None
            active_name = "upload.jpg"

            if uploaded_file is not None:
                active_bytes = uploaded_file.getvalue()
                active_name = uploaded_file.name
            elif camera_file is not None:
                active_bytes = camera_file.getvalue()
                active_name = "camera.jpg"
            elif "loaded_sample_bytes" in st.session_state:
                active_bytes = st.session_state["loaded_sample_bytes"]
                active_name = st.session_state.get("loaded_sample_name", "sample.jpg")

        with col_view:
            st.markdown(f"### 🖼️ {get_text('preview_title', lang)}")

            if active_bytes is not None:
                is_valid, img, err_msg = validate_image_upload(active_bytes, active_name)

                if not is_valid:
                    st.error(f"❌ Upload Error: {err_msg}")
                else:
                    # Immediate high-resolution thumbnail preview
                    st.image(
                        img,
                        caption=f"🌿 {get_text('orig_image', lang)} ({active_name})",
                        use_container_width=True,
                    )
                    st.success(f"✅ {get_text('preview_ready', lang)}")

                    # Primary Diagnose Button
                    if st.button(get_text("diagnose_btn", lang), type="primary", use_container_width=True):
                        with st.spinner(get_text("analyzing_progress", lang)):
                            model, device = load_cached_model(model_choice)
                            overlay, heatmap, pred_idx, confidence = generate_explanation(
                                model=model,
                                image=img,
                                model_name=model_choice,
                                device=device,
                            )

                            pred_raw_class = PLANT_VILLAGE_CLASSES[pred_idx] if pred_idx < len(PLANT_VILLAGE_CLASSES) else "Unknown"
                            plant_name, disease_name = format_class_name(pred_raw_class)
                            is_uncertain = confidence < conf_threshold

                            # Log record to SQLite
                            img_hash = compute_image_hash(img)
                            log_diagnosis_to_history(
                                image_hash=img_hash,
                                plant_name=plant_name,
                                predicted_disease=disease_name,
                                confidence=confidence,
                                model_name=model_choice,
                                flagged_uncertain=is_uncertain,
                            )

                            # Save to session state
                            st.session_state["last_result"] = {
                                "img": img,
                                "overlay": overlay,
                                "heatmap": heatmap,
                                "pred_class": pred_raw_class,
                                "plant_name": plant_name,
                                "disease_name": disease_name,
                                "confidence": confidence,
                                "is_uncertain": is_uncertain,
                                "model_choice": model_choice,
                            }
            else:
                st.info(f"💡 {get_text('history_empty', lang)}")

        # ----------------------------------------------------------------------
        # RESULTS PRESENTATION CARD & STRUCTURED TABS
        # ----------------------------------------------------------------------
        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            st.markdown("---")
            st.markdown(f"## 📋 {get_text('results_title', lang)}")

            # Prominent Low-Confidence Safety Alert Banner if uncertain
            if res["is_uncertain"]:
                render_safety_alert(lang)

            # High-Impact Clinical Result Card
            card_class = "diagnosis-result-card uncertain" if res["is_uncertain"] else "diagnosis-result-card"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <span style="font-size: 0.88rem; color: #15803D; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; background: #DCFCE7; padding: 4px 10px; border-radius: 6px; border: 1px solid #86EFAC;">
                                🌱 {get_text('plant_species', lang)}: <strong>{res['plant_name']}</strong>
                            </span>
                            <h2 style="color: #064E3B; margin: 8px 0 6px 0; font-size: 1.85rem; font-family: 'Outfit', sans-serif;">
                                {res['disease_name']}
                            </h2>
                            <div style="font-size: 0.85rem; color: #64748B;">
                                AI Architecture: <code>{res['model_choice']}</code> &nbsp;|&nbsp; Status: <strong>{'⚠️ Low Confidence Review' if res['is_uncertain'] else '✅ Verified Prediction'}</strong>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Visual Confidence Gauge with accessible color-paired label
            render_confidence_indicator(res["confidence"], threshold=conf_threshold, lang=lang)

            # ------------------------------------------------------------------
            # STRUCTURED DRILL-DOWN TABS (XAI image always visible in Tab 1)
            # ------------------------------------------------------------------
            res_tab_xai, res_tab_symptoms, res_tab_prev, res_tab_treat, res_tab_cite = st.tabs([
                get_text("tab_xai", lang),
                get_text("tab_symptoms", lang),
                get_text("tab_prevention", lang),
                get_text("tab_treatment", lang),
                get_text("tab_citations", lang),
            ])

            rec = get_recommendation_for_disease(res["pred_class"])

            # TAB 1: EXPLAINABLE AI VISUAL ATTENTION (VISIBLE BY DEFAULT)
            with res_tab_xai:
                st.markdown(f"### {get_text('xai_title', lang)}")
                st.info(get_text("xai_desc", lang))

                col_x1, col_x2 = st.columns(2, gap="medium")
                with col_x1:
                    st.image(
                        res["img"],
                        caption=f"📷 {get_text('orig_image', lang)}",
                        use_container_width=True,
                    )
                with col_x2:
                    st.image(
                        res["overlay"],
                        caption=f"🔬 {get_text('xai_overlay', lang)} ({res['model_choice']})",
                        use_container_width=True,
                    )

                # Interactive 3D Saliency Model (Stretch Goal)
                st.markdown("#### 🌿 3D Specimen Lesion Attention Spotlight")
                render_3d_xai_leaf(
                    confidence=res["confidence"],
                    disease_name=res["disease_name"],
                    height=280,
                )

            # TAB 2: SYMPTOMS & PATHOLOGY
            with res_tab_symptoms:
                st.markdown(f"### {get_text('symptoms', lang)}")
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 22px; font-size: 1.02rem; line-height: 1.6; color: #14281D; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
                        🔍 <strong>Clinical Diagnostic Indicators:</strong><br><br>
                        {rec.get('symptoms', 'No specific symptoms recorded for this class.')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # TAB 3: PREVENTION & CULTURAL CONTROL
            with res_tab_prev:
                st.markdown(f"### {get_text('prevention', lang)}")
                st.markdown(
                    f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 14px; padding: 22px; font-size: 1.02rem; line-height: 1.6; color: #14532D; box-shadow: 0 2px 8px rgba(22, 163, 74, 0.04);">
                        🛡️ <strong>Recommended Cultural & Preventative Practices:</strong><br><br>
                        {rec.get('prevention', 'General crop sanitation and canopy spacing recommended.')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # TAB 4: TREATMENT & MANAGEMENT
            with res_tab_treat:
                st.markdown(f"### {get_text('treatment', lang)}")
                st.markdown(
                    f"""
                    <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 14px; padding: 22px; font-size: 1.02rem; line-height: 1.6; color: #92400E; box-shadow: 0 2px 8px rgba(217, 119, 6, 0.04);">
                        💊 <strong>Agronomic Guidance & Product Recommendations:</strong><br><br>
                        {rec.get('treatment', 'Consult extension officer for approved local fungicides and dosage rates.')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # TAB 5: CITATIONS & SOURCES
            with res_tab_cite:
                st.markdown(f"### {get_text('source', lang)}")
                st.success(f"📖 {rec.get('source_citation', 'Extension Guidance')}")
                if rec.get("needs_review"):
                    st.warning("⚠️ *Note: This class entry is flagged for specialist domain review.*")
                if rec.get("last_reviewed"):
                    st.caption(f"Last domain verification date: {rec.get('last_reviewed')}")

            # Downloadable Summary Report
            report_text = f"""=======================================================
PLANTCARE AI - CLINICAL DIAGNOSTIC REPORT
=======================================================
Plant Species: {res['plant_name']}
Diagnosis: {res['disease_name']}
Confidence: {res['confidence'] * 100:.2f}%
Model Architecture: {res['model_choice']}
Safety Threshold: {conf_threshold * 100:.0f}%
Flagged Uncertain: {'Yes' if res['is_uncertain'] else 'No'}

SYMPTOMS:
{rec.get('symptoms', 'N/A')}

PREVENTION:
{rec.get('prevention', 'N/A')}

TREATMENT GUIDANCE:
{rec.get('treatment', 'N/A')}

EXTENSION CITATION:
{rec.get('source_citation', 'N/A')}
=======================================================
"""
            st.download_button(
                label=f"📄 {get_text('download_report_btn', lang)}",
                data=report_text,
                file_name=f"PlantCare_Diagnosis_{res['plant_name']}_{res['disease_name']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ==========================================================================
    # TAB 2: RESEARCH BENCHMARKS & ROBUSTNESS
    # ==========================================================================
    with tab_metrics:
        st.markdown("### 📊 Benchmark Metrics & Robustness Analysis")
        comp_csv = METRICS_DIR / "comparison_table.csv"

        if comp_csv.exists():
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("Evaluation metrics table not yet generated. Run `python -m src.evaluate` to populate.")

        # Result charts
        chart_f1 = GRAPHS_DIR / "accuracy_f1_comparison.png"
        chart_scatter = GRAPHS_DIR / "latency_vs_f1_scatter.png"
        chart_calib = GRAPHS_DIR / "reliability_calibration_diagram.png"

        if chart_f1.exists():
            st.image(str(chart_f1), caption=get_text("accuracy_f1_chart", lang), use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        if chart_scatter.exists():
            with col_g1:
                st.image(str(chart_scatter), caption=get_text("latency_chart", lang), use_container_width=True)
        if chart_calib.exists():
            with col_g2:
                st.image(str(chart_calib), caption=get_text("calibration_chart", lang), use_container_width=True)

    # ==========================================================================
    # TAB 3: DIAGNOSIS HISTORY GALLERY
    # ==========================================================================
    with tab_history:
        st.markdown(f"### 📜 {get_text('recent_history', lang)}")
        history_records = get_recent_diagnoses(limit=30)
        render_history_gallery(history_records, lang=lang)


if __name__ == "__main__":
    main()
