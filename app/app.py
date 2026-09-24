"""PlantCare AI - Interactive Streamlit Web Application with Explainable AI & Calm Botanical Design."""

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
    get_dataset_catalog_summary,
)

try:
    from app.components import (
        get_text,
        inject_custom_styles,
        render_compact_hero,
        render_workflow_stepper,
        render_footer_disclaimer,
        render_3d_plant_hero,
        render_3d_xai_leaf,
        render_confidence_indicator,
        render_safety_alert,
        render_history_gallery,
    )
except (ImportError, ModuleNotFoundError):
    from components import (
        get_text,
        inject_custom_styles,
        render_compact_hero,
        render_workflow_stepper,
        render_footer_disclaimer,
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
        "emoji": "🍅",
        "title": "Tomato Specimen",
        "disease": "Early Blight",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00000_sample_0005.jpg",
        "species": "Tomato",
    },
    {
        "emoji": "🍎",
        "title": "Apple Specimen",
        "disease": "Apple Scab",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00001_sample_0008.jpg",
        "species": "Apple",
    },
    {
        "emoji": "🌽",
        "title": "Corn Foliage",
        "disease": "Common Rust",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00002_sample_0004.jpg",
        "species": "Corn (maize)",
    },
    {
        "emoji": "🍇",
        "title": "Grape Leaf",
        "disease": "Black Rot",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00003_sample_0010.jpg",
        "species": "Grape",
    },
    {
        "emoji": "🫑",
        "title": "Pepper Foliage",
        "disease": "Bacterial Spot",
        "file": PROJECT_ROOT / "data" / "robustness_test_set" / "blur_mild" / "deg_00004_sample_0003.jpg",
        "species": "Pepper, bell",
    },
]


def main():
    # --------------------------------------------------------------------------
    # SIDEBAR: Clean, Grouped Configuration & System Settings
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid #E5E9E2;">
                <div style="font-size: 1.8rem; background: #EBF2E8; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid #BBF7D0;">🌿</div>
                <div>
                    <div style="font-weight: 800; font-size: 1.18rem; color: #064E3B; line-height: 1.1;">PlantCare AI</div>
                    <div style="font-size: 0.74rem; color: #059669; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Clinical XAI Core</div>
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

        st.markdown(f"#### ⚙️ {get_text('select_model', lang)}")
        model_choice = st.selectbox(
            get_text("select_model", lang),
            ["resnet50", "efficientnet_b0", "vit_b_16"],
            format_func=lambda x: {
                "resnet50": "ResNet50 (Deep CNN + Grad-CAM)",
                "efficientnet_b0": "EfficientNet-B0 (Lightweight CNN)",
                "vit_b_16": "ViT-B/16 (Vision Transformer + Rollout)",
            }.get(x, x),
            label_visibility="collapsed",
        )

        st.markdown(f"#### 🛡️ {get_text('confidence_threshold', lang)}")
        threshold_val = st.slider(
            get_text("confidence_threshold", lang),
            min_value=30,
            max_value=90,
            value=int(CONFIG.get("app", {}).get("confidence_threshold", 0.60) * 100),
            step=5,
            help=get_text("confidence_threshold_help", lang),
            label_visibility="collapsed",
        )
        conf_threshold = threshold_val / 100.0

        st.markdown("---")
        with st.expander("ℹ️ System & Research Metadata", expanded=False):
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #4A5E51; line-height: 1.5;">
                    <strong style="color: #064E3B;">🎓 Academic Research Prototype</strong><br>
                    Department of Computer Science & Engineering<br>
                    <em>Chandigarh University</em><br>
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed #E5E9E2;">
                        • <strong>Dataset:</strong> ~100,000 Cataloged<br>
                        • <strong>Classes:</strong> 38 Pathologies<br>
                        • <strong>Explainability:</strong> Grad-CAM & Attention Rollout<br>
                        • <strong>Knowledge Base:</strong> FAO / ICAR SQLite
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # COMPACT HERO BANNER & HORIZONTAL STEPPER
    # --------------------------------------------------------------------------
    device = get_device()
    render_compact_hero(
        model_name=model_choice,
        confidence_threshold=conf_threshold,
        device=str(device),
        lang=lang,
    )

    render_workflow_stepper(current_step=1, lang=lang)

    # --------------------------------------------------------------------------
    # MAIN TABS: Diagnosis & XAI | Research Metrics | Diagnosis History | Big Data DB
    # --------------------------------------------------------------------------
    tab_diag, tab_metrics, tab_history, tab_db = st.tabs([
        "🔬 Diagnosis & XAI",
        "📊 Research Metrics & Robustness",
        "📜 Diagnosis History",
        "🗄️ Big Data 100K Catalog & DB",
    ])

    # ==========================================================================
    # TAB 1: DIAGNOSIS & EXPLAINABLE AI
    # ==========================================================================
    with tab_diag:
        # Attractive Quick-Test Specimen Cards
        st.markdown(
            f"""
            <div class="specimen-card-box">
                <div class="specimen-card-header">
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem; color: #132A1C;">⚡ {get_text('sample_tray_title', lang)}</div>
                        <div style="font-size: 0.82rem; color: #7D9285;">{get_text('sample_tray_hint', lang)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        cols = [col_s1, col_s2, col_s3, col_s4, col_s5]

        for i, spec in enumerate(SAMPLE_SPECIMENS):
            with cols[i]:
                btn_label = f"{spec['emoji']} {spec['title']}\n({spec['disease']})"
                if st.button(btn_label, key=f"quick_sample_{i}", use_container_width=True):
                    if spec["file"].exists():
                        with open(spec["file"], "rb") as f:
                            st.session_state["loaded_sample_bytes"] = f.read()
                            st.session_state["loaded_sample_name"] = spec["file"].name
                            st.session_state["selected_sample_species"] = spec["species"]

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

        # Upload and Camera Input Section
        col_input, col_view = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown(f"### 📤 {get_text('upload_label', lang)}")

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

            # File uploader
            uploaded_file = st.file_uploader(
                get_text("upload_label", lang),
                type=["jpg", "jpeg", "png", "webp"],
                help=get_text("upload_help", lang),
                key="leaf_file_uploader",
                label_visibility="collapsed",
            )

            # Camera input
            camera_img = st.camera_input(
                get_text("camera_label", lang),
                key="leaf_camera_input",
                label_visibility="collapsed",
            )

            raw_bytes: Optional[bytes] = None
            filename: str = ""

            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                filename = uploaded_file.name
            elif camera_img is not None:
                raw_bytes = camera_img.getvalue()
                filename = "camera_capture.jpg"
            elif "loaded_sample_bytes" in st.session_state:
                raw_bytes = st.session_state["loaded_sample_bytes"]
                filename = st.session_state.get("loaded_sample_name", "specimen.jpg")

        # Leaf Preview & Primary Action Button
        with col_view:
            st.markdown(f"### 🖼️ {get_text('preview_title', lang)}")

            if raw_bytes is not None:
                is_valid, img, err = validate_image_upload(raw_bytes, filename)
                if not is_valid:
                    st.error(f"❌ {err}")
                else:
                    st.image(img, caption=get_text("preview_ready", lang), use_container_width=True)

                    if st.button(
                        f"⚡ {get_text('diagnose_btn', lang)}",
                        key="btn_run_diagnosis",
                        type="primary",
                        use_container_width=True,
                    ):
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

            # High-Impact Calm Botanical Result Card
            card_accent = "#DC2626" if res["is_uncertain"] else "#059669"
            st.markdown(
                f"""
                <div class="botanical-card" style="border-left: 5px solid {card_accent};">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <span style="font-size: 0.78rem; color: #047857; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; background: #DCFCE7; padding: 4px 12px; border-radius: 9999px; border: 1px solid #86EFAC;">
                                🌱 {get_text('plant_species', lang)}: <strong>{res['plant_name']}</strong>
                            </span>
                            <h2 style="color: #132A1C; margin: 10px 0 6px 0; font-size: 1.95rem; font-weight: 800; letter-spacing: -0.02em;">
                                {res['disease_name']}
                            </h2>
                            <div style="font-size: 0.84rem; color: #4A5E51;">
                                Neural Architecture: <code>{res['model_choice']}</code> &nbsp;|&nbsp; Status: <strong>{'⚠️ Low Confidence Review' if res['is_uncertain'] else '✅ High Confidence Verified'}</strong>
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
            # STRUCTURED DRILL-DOWN TABS
            # ------------------------------------------------------------------
            res_tab_xai, res_tab_symptoms, res_tab_prev, res_tab_treat, res_tab_cite = st.tabs([
                get_text("tab_xai", lang),
                get_text("tab_symptoms", lang),
                get_text("tab_prevention", lang),
                get_text("tab_treatment", lang),
                get_text("tab_citations", lang),
            ])

            rec = get_recommendation_for_disease(res["pred_class"])

            # TAB 1: EXPLAINABLE AI VISUAL ATTENTION
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

                # Interactive 3D Saliency Model
                st.markdown("#### 🌿 3D Specimen Lesion Attention Spotlight")
                render_3d_xai_leaf(
                    confidence=res["confidence"],
                    disease_name=res["disease_name"],
                    height=280,
                )

            # TAB 2: CLINICAL SYMPTOMS
            with res_tab_symptoms:
                st.markdown(f"### {get_text('symptoms', lang)}")
                st.markdown(
                    f"""
                    <div class="botanical-card" style="font-size: 0.98rem; line-height: 1.65; color: #132A1C;">
                        <div style="font-weight: 700; color: #059669; margin-bottom: 8px;">🔍 Pathological Symptoms Profile:</div>
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
                    <div class="botanical-card" style="font-size: 0.98rem; line-height: 1.65; color: #064E3B; background: #F0FDF4; border-color: #BBF7D0;">
                        <div style="font-weight: 700; color: #047857; margin-bottom: 8px;">🛡️ Recommended Cultural & Preventative Protocol:</div>
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
                    <div class="botanical-card" style="font-size: 0.98rem; line-height: 1.65; color: #92400E; background: #FEF3C7; border-color: #FDE68A;">
                        <div style="font-weight: 700; color: #B45309; margin-bottom: 8px;">💊 Agronomic Guidance & Product Recommendations:</div>
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

    # ==========================================================================
    # TAB 4: BIG DATA 100K IMAGE CATALOG & KNOWLEDGE BASE DATABASE
    # ==========================================================================
    with tab_db:
        st.markdown("### 🗄️ Big Data 100,000 Image Catalog & Knowledge Base")
        st.caption("SQLite High-Throughput Relational Storage with Indexed Partitions & Provenance Tracking")

        db_stats = get_dataset_catalog_summary()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Cataloged Images", f"{db_stats.get('total_images', 0):,}")
        with c2:
            st.metric("Disease Classes", f"{db_stats.get('total_classes', 0)}")
        with c3:
            st.metric("Plant Species", f"{db_stats.get('total_species', 0)}")
        with c4:
            st.metric("Dataset Footprint", f"{db_stats.get('total_dataset_gb', 0):.2f} GB")

        splits = db_stats.get("splits", {})
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Train Split (70%)", f"{splits.get('train', 0):,} images")
        with col_s2:
            st.metric("Val Split (15%)", f"{splits.get('val', 0):,} images")
        with col_s3:
            st.metric("Test Split (15%)", f"{splits.get('test', 0):,} images")

        st.markdown("---")
        st.markdown("#### 🔍 Interactive Dataset Catalog Inspector")
        try:
            import sqlite3
            conn = sqlite3.connect(str(DB_PATH))
            preview_df = pd.read_sql_query(
                "SELECT image_uid, file_path, dataset_source, plant_name, disease_class, split_type, is_augmented, augmentation_type, file_size_kb, quality_score FROM image_dataset LIMIT 25",
                conn,
            )
            st.dataframe(preview_df, use_container_width=True)
            conn.close()
        except Exception as err:
            st.warning(f"Could not load image catalog preview: {err}")

    # --------------------------------------------------------------------------
    # SUBTLE FOOTER DISCLAIMER BANNER
    # --------------------------------------------------------------------------
    render_footer_disclaimer(lang=lang)


if __name__ == "__main__":
    main()
