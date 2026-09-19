"""PlantCare AI - Advanced UI components, 3D Three.js visualizers, design system, and localization."""

from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. LOCALIZATION & TEXT STRINGS (English & Hindi)
# ==============================================================================

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_badge": "XAI Botanical Intelligence v2.0",
        "title": "PlantCare AI",
        "title_full": "🌿 PlantCare AI — Explainable Plant Disease Diagnosis",
        "subtitle": "Clinical-grade crop pathology detection with deep neural networks & visual explainability (XAI)",
        "hero_value_prop": "Instant, Explainable Botanical Disease Diagnosis for Healthy Crops",
        "hero_how_it_works": "1. Upload Photo → 2. Deep AI Diagnosis → 3. Grad-CAM Explanation → 4. Agronomic Guidance",
        "step_1": "1. Upload Photo",
        "step_2": "2. AI Analysis",
        "step_3": "3. XAI Heatmap",
        "step_4": "4. Treatment Plan",
        "disclaimer": "⚠️ Research Prototype Disclaimer: This system is an academic decision-support tool (Chandigarh University) and not a replacement for certified on-field agronomist consultation.",
        "sidebar_title": "Configuration & Settings",
        "select_model": "Select AI Model Architecture",
        "confidence_threshold": "Uncertainty Safety Threshold (%)",
        "confidence_threshold_help": "Predictions below this threshold are flagged as uncertain to prevent misapplication of chemical treatments.",
        "select_plant": "Select Crop / Plant Species",
        "upload_label": "Upload Leaf Photograph",
        "upload_help": "Supported formats: JPG, JPEG, PNG, WebP (Max 10MB)",
        "upload_drag_hint": "Drag and drop leaf image here, or click to browse",
        "camera_label": "Or Capture Photo via Camera",
        "diagnose_btn": "🔍 Analyze Leaf Health",
        "analyzing_progress": "Analyzing leaf pathology & computing Grad-CAM visual attention...",
        "preview_title": "Selected Leaf Preview",
        "preview_ready": "Image loaded successfully. Click 'Analyze Leaf Health' to begin diagnosis.",
        "results_title": "Diagnosis Results & Clinical Report",
        "plant_species": "Plant Species",
        "prediction": "Identified Disease",
        "confidence": "Diagnosis Confidence",
        "confidence_band_high": "High Confidence",
        "confidence_band_moderate": "Moderate Confidence",
        "confidence_band_low": "Uncertain — Verification Required",
        "uncertain_warning_title": "⚠️ Low Confidence Diagnosis Alert",
        "uncertain_warning": "Prediction confidence is below the safety threshold. Do not apply intensive chemical treatments without consulting an agricultural extension specialist.",
        "tab_xai": "🔬 Explainable AI (XAI)",
        "tab_symptoms": "🔍 Symptoms",
        "tab_prevention": "🛡️ Prevention",
        "tab_treatment": "💊 Treatment",
        "tab_citations": "📚 Verified Sources",
        "tab_3d_view": "🌿 3D Pathology Visualizer",
        "xai_title": "Visual Saliency & Attention Map",
        "xai_desc": "The highlighted regions (warm colors: red, orange, yellow) indicate the exact leaf lesions and anatomical features that guided the AI's diagnosis.",
        "orig_image": "Original Input Leaf",
        "xai_overlay": "Grad-CAM Attention Overlay",
        "symptoms": "Pathological Symptoms",
        "prevention": "Preventative & Cultural Measures",
        "treatment": "Agronomic & Treatment Guidance",
        "source": "Verified Extension Citation",
        "recent_history": "Recent Diagnosis History",
        "history_empty": "No leaf diagnoses recorded yet. Upload a leaf image above to start.",
        "model_comparison": "Benchmark Metrics & Robustness",
        "clear_history": "Clear History",
        "three_d_hero_caption": "Interactive 3D Botanical Specimen (Drag to rotate, mouse parallax active)",
        "three_d_xai_caption": "3D Interactive Leaf Specimen with Projected Attention Field",
        "webgl_fallback": "Botanical Vision Core",
        "accuracy_f1_chart": "F1-Score Comparison under Clean vs Degraded Robustness Conditions",
        "latency_chart": "Inference Latency vs Macro F1 Trade-off",
        "calibration_chart": "Confidence Reliability Diagrams",
        "view_details": "View Details",
        "flagged_uncertain_badge": "Safety Flagged",
        "flagged_confident_badge": "Verified",
        "sample_tray_title": "⚡ Quick Test: Select Preloaded Leaf Specimen",
        "sample_tray_hint": "Click any specimen to load and test instantly without file upload",
        "top_predictions_title": "Probable Pathogen Distribution",
        "download_report_btn": "📥 Download Clinical Summary Report",
    },
    "hi": {
        "app_badge": "व्याख्यात्मक वनस्पति एआई v2.0",
        "title": "प्लांटकेयर एआई",
        "title_full": "🌿 PlantCare AI — व्याख्यात्मक पादप रोग निदान",
        "subtitle": "डीप न्यूरल नेटवर्क और व्याख्यात्मक एआई (XAI) द्वारा संचालित सटीक पादप विकृति विज्ञान",
        "hero_value_prop": "स्वस्थ फसलों के लिए तत्काल, व्याख्यात्मक पादप रोग निदान",
        "hero_how_it_works": "1. पत्ती की तस्वीर अपलोड करें → 2. एआई द्वारा रोग की पहचान → 3. XAI हीटमैप स्पष्टीकरण → 4. साक्ष्य-आधारित कृषि मार्गदर्शन",
        "step_1": "1. फोटो अपलोड",
        "step_2": "2. एआई विश्लेषण",
        "step_3": "3. XAI हीटमैप",
        "step_4": "4. उपचार योजना",
        "disclaimer": "⚠️ अनुसंधान प्रोटोटाइप अस्वीकरण: यह प्रणाली एक शोध निर्णय-सहायता उपकरण (चंडीगढ़ विश्वविद्यालय) है और पेशेवर कृषि विशेषज्ञ परामर्श का विकल्प नहीं है।",
        "sidebar_title": "कॉन्फ़िगरेशन एवं सेटिंग्स",
        "select_model": "एआई मॉडल आर्किटेक्चर चुनें",
        "confidence_threshold": "अनिश्चितता सुरक्षा सीमा (%)",
        "confidence_threshold_help": "इस सीमा से कम विश्वास वाले निदानों को अनिश्चित चिह्नित किया जाता है ताकि गलत उपचार से बचा जा सके।",
        "select_plant": "फसल / पौधे की प्रजाति चुनें",
        "upload_label": "पत्ती की तस्वीर अपलोड करें",
        "upload_help": "स्वीकृत प्रारूप: JPG, JPEG, PNG, WebP (अधिकतम 10MB)",
        "upload_drag_hint": "पत्ती की छवि यहाँ खींचकर छोड़ें, या ब्राउज़ करने के लिए क्लिक करें",
        "camera_label": "या कैमरे से तस्वीर खींचें",
        "diagnose_btn": "🔍 पत्ती के स्वास्थ्य की जांच करें",
        "analyzing_progress": "पत्ती के विकृति विज्ञान का विश्लेषण और Grad-CAM दृश्य ध्यान की गणना की जा रही है...",
        "preview_title": "चयनित पत्ती का पूर्वावलोकन",
        "preview_ready": "तस्वीर सफलतापूर्वक लोड हो गई। निदान शुरू करने के लिए 'जांच करें' पर क्लिक करें।",
        "results_title": "निदान परिणाम एवं क्लिनिकल रिपोर्ट",
        "plant_species": "पौधे की प्रजाति",
        "prediction": "पहचाना गया रोग",
        "confidence": "निदान विश्वास (कॉन्फिडेंस)",
        "confidence_band_high": "उच्च विश्वास (High)",
        "confidence_band_moderate": "मध्यम विश्वास (Moderate)",
        "confidence_band_low": "अनिश्चित — विशेषज्ञ सत्यापन आवश्यक",
        "uncertain_warning_title": "⚠️ कम विश्वास निदान चेतावनी",
        "uncertain_warning": "निदान का विश्वास सुरक्षा सीमा से कम है। किसी कृषि विस्तार विशेषज्ञ से परामर्श किए बिना गहन रासायनिक उपचार न करें।",
        "tab_xai": "🔬 व्याख्यात्मक एआई (XAI)",
        "tab_symptoms": "🔍 रोग के लक्षण",
        "tab_prevention": "🛡️ रोकथाम के उपाय",
        "tab_treatment": "💊 उपचार एवं प्रबंधन",
        "tab_citations": "📚 सत्यापित स्रोत",
        "tab_3d_view": "🌿 3D पैथोलॉजी विज़ुअलाइज़र",
        "xai_title": "दृश्य ध्यान एवं स्पष्टीकरण (Grad-CAM)",
        "xai_desc": "उजागर किए गए क्षेत्र (लाल/पीले रंग) उन प्रमुख पत्ती के हिस्सों को दर्शाते हैं जिन्होंने मॉडल के निर्णय को निर्देशित किया।",
        "orig_image": "मूल इनपुट तस्वीर",
        "xai_overlay": "Grad-CAM ध्यान ओवरले",
        "symptoms": "रोग के लक्षण",
        "prevention": "रोकथाम एवं सांस्कृतिक उपाय",
        "treatment": "कृषि एवं उपचार दिशानिर्देश",
        "source": "सत्यापित कृषि विस्तार स्रोत",
        "recent_history": "हालिया निदान इतिहास",
        "history_empty": "अभी तक कोई निदान दर्ज नहीं हुआ है। शुरू करने के लिए ऊपर एक तस्वीर अपलोड करें।",
        "model_comparison": "मॉडल तुलना और मेट्रिक्स",
        "clear_history": "इतिहास साफ़ करें",
        "three_d_hero_caption": "इंटरैक्टिव 3D वनस्पति मॉडल (झुकाने के लिए खींचें / होवर करें)",
        "three_d_xai_caption": "प्रक्षेपित ध्यान क्षेत्र के साथ 3D पत्ती नमूना",
        "webgl_fallback": "वानस्पतिक दृष्टि कोर",
        "accuracy_f1_chart": "स्वच्छ बनाम विकृत परिस्थितियों में F1-स्कोर तुलना",
        "latency_chart": "इनफ़्रेंस विलंबता बनाम मैक्रो F1 संतुलन",
        "calibration_chart": "विश्वसनीयता अंशांकन आरेख",
        "view_details": "विवरण देखें",
        "flagged_uncertain_badge": "सुरक्षा ध्वज",
        "flagged_confident_badge": "सत्यापित",
        "sample_tray_title": "⚡ त्वरित परीक्षण: पूर्व-लोड नमूना पत्ती चुनें",
        "sample_tray_hint": "बिना फ़ाइल अपलोड किए तुरंत परीक्षण करने के लिए किसी भी नमूने पर क्लिक करें",
        "top_predictions_title": "संभावित रोगज़नक़ वितरण",
        "download_report_btn": "📥 क्लिनिकल सारांश रिपोर्ट डाउनलोड करें",
    },
}


def get_text(key: str, lang: str = "en") -> str:
    """Returns localized string with fallback to English."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ==============================================================================
# 2. DESIGN SYSTEM & CSS INJECTION (Modern Botanical Aesthetics)
# ==============================================================================

def inject_custom_styles():
    """Injects high-contrast, responsive typography, glassmorphism, and micro-interactions."""
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --color-primary: #16A34A;
            --color-primary-dark: #15803D;
            --color-primary-deep: #064E3B;
            --color-primary-light: #DCFCE7;
            --color-accent-amber: #D97706;
            --color-accent-blue: #0284C7;
            --color-surface-glass: rgba(255, 255, 255, 0.88);
            --color-bg-warm: #F8FAF6;
            --color-border-subtle: #E2E8F0;
            --color-border-emerald: #BBF7D0;
            --color-text-main: #14281D;
            --color-text-muted: #4B6354;
            --color-success: #16A34A;
            --color-warning: #D97706;
            --color-danger: #DC2626;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --shadow-glass: 0 10px 30px -4px rgba(22, 163, 74, 0.07), 0 4px 12px -2px rgba(20, 40, 29, 0.04);
            --shadow-float: 0 20px 40px -10px rgba(6, 78, 59, 0.12);
        }

        /* Global Typography */
        html, body, [class*="css"], .stMarkdown, p, div, span, label {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--color-text-main);
        }

        h1, h2, h3, h4, h5, h6, .stTitle, [data-testid="stHeader"] {
            font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            color: var(--color-text-main) !important;
            letter-spacing: -0.025em;
        }

        /* Glassmorphic Hero Container */
        .plant-hero-container {
            background: linear-gradient(135deg, rgba(240, 253, 244, 0.95) 0%, rgba(248, 250, 246, 0.9) 50%, rgba(236, 253, 245, 0.95) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--color-border-emerald);
            border-radius: var(--radius-lg);
            padding: 26px 32px;
            margin-bottom: 20px;
            box-shadow: var(--shadow-glass);
            position: relative;
            overflow: hidden;
        }

        .plant-hero-container::before {
            content: "";
            position: absolute;
            top: -50px;
            right: -50px;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(134, 239, 172, 0.35) 0%, rgba(248, 250, 246, 0) 70%);
            pointer-events: none;
        }

        .plant-hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #DCFCE7;
            color: #15803D;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 5px 14px;
            border-radius: 9999px;
            margin-bottom: 12px;
            border: 1px solid #86EFAC;
            box-shadow: 0 1px 3px rgba(22, 163, 74, 0.1);
        }

        .plant-hero-title {
            font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.15;
            color: #064E3B !important;
            margin-bottom: 8px;
            letter-spacing: -0.03em;
        }

        .plant-hero-desc {
            font-size: 1.05rem;
            color: #2D6A4F;
            margin-bottom: 20px;
            max-width: 840px;
            line-height: 1.55;
        }

        /* 4-Step Workflow Badges */
        .workflow-steps-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 16px;
        }

        .workflow-step {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #D1FAE5;
            border-radius: var(--radius-md);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.88rem;
            font-weight: 600;
            color: #14532D;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .workflow-step:hover {
            border-color: #10B981;
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(22, 163, 74, 0.12);
            background: #FFFFFF;
        }

        .workflow-step-icon {
            width: 28px;
            height: 28px;
            background: #ECFDF5;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.88rem;
            color: #059669;
            flex-shrink: 0;
            border: 1px solid #A7F3D0;
        }

        /* Quick Sample Leaf Selection Tray */
        .sample-tray-container {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin: 14px 0 20px 0;
            box-shadow: var(--shadow-glass);
        }

        .sample-tray-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #064E3B;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Modernized Uploader Area */
        [data-testid="stFileUploader"] {
            border-radius: var(--radius-md);
            padding: 2px;
        }

        [data-testid="stFileUploader"] section {
            border: 2px dashed #86EFAC !important;
            background-color: #F8FAF6 !important;
            border-radius: var(--radius-md) !important;
            padding: 24px 16px !important;
            transition: all 0.25s ease;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: #16A34A !important;
            background-color: #F0FDF4 !important;
        }

        /* Primary Action Buttons */
        .stButton button {
            border-radius: var(--radius-md) !important;
            font-weight: 700 !important;
            padding: 12px 28px !important;
            min-height: 48px !important;
            font-size: 1rem !important;
            letter-spacing: -0.01em !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 4px 12px rgba(22, 163, 74, 0.18) !important;
        }

        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(22, 163, 74, 0.28) !important;
        }

        /* Result Clinical Card */
        .diagnosis-result-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FBF8 100%);
            border: 1px solid #BBF7D0;
            border-radius: var(--radius-lg);
            padding: 26px 30px;
            box-shadow: var(--shadow-float);
            margin-top: 16px;
            margin-bottom: 24px;
            border-left: 8px solid #16A34A;
            position: relative;
        }

        .diagnosis-result-card.uncertain {
            border-left-color: #DC2626;
            border-color: #FECACA;
            background: linear-gradient(135deg, #FFFFFF 0%, #FFFDFD 100%);
        }

        /* Confidence Badges */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .badge-high {
            background-color: #DCFCE7;
            color: #15803D;
            border: 1px solid #86EFAC;
        }

        .badge-mod {
            background-color: #FEF3C7;
            color: #B45309;
            border: 1px solid #FCD34D;
        }

        .badge-low {
            background-color: #FEE2E2;
            color: #B91C1C;
            border: 1px solid #FCA5A5;
        }

        /* Uncertainty Safety Callout */
        .safety-alert-banner {
            background: linear-gradient(135deg, #FEF2F2 0%, #FFF1F2 100%);
            border: 1px solid #FECACA;
            border-left: 6px solid #DC2626;
            border-radius: var(--radius-md);
            padding: 18px 22px;
            margin: 18px 0;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.08);
        }

        .safety-alert-title {
            color: #991B1B;
            font-weight: 800;
            font-size: 1.05rem;
            margin-bottom: 4px;
        }

        .safety-alert-text {
            color: #7F1D1D;
            font-size: 0.94rem;
            line-height: 1.5;
            margin: 0;
        }

        /* History Gallery Cards */
        .history-gallery-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: var(--shadow-glass);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .history-gallery-card:hover {
            border-color: #86EFAC;
            box-shadow: 0 8px 18px rgba(22, 163, 74, 0.1);
            transform: translateY(-2px);
        }

        /* Sidebar Frosted Card */
        [data-testid="stSidebar"] {
            background-color: #F8FAF6 !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Streamlit Tabs Polish */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #EBF2E8;
            padding: 6px;
            border-radius: var(--radius-md);
            margin-bottom: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-sm) !important;
            padding: 10px 18px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: #2D6A4F !important;
            background: transparent !important;
            border: none !important;
            transition: all 0.2s ease !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #064E3B !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            font-weight: 700 !important;
        }

        /* Mobile Viewport Responsiveness */
        @media screen and (max-width: 768px) {
            .workflow-steps-container {
                grid-template-columns: 1fr 1fr;
            }
            .plant-hero-title {
                font-size: 1.6rem;
            }
            .plant-hero-container {
                padding: 20px 16px;
            }
            /* Stack Streamlit Columns Vertically on Small Viewports */
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. 3D THREE.JS HERO COMPONENT (Part B - Bioluminescent Specimen)
# ==============================================================================

def render_3d_plant_hero(height: int = 340):
    """
    Renders an interactive, procedural 3D botanical specimen using Three.js inside an iframe.
    - Zero external 3D asset downloads (pure procedural geometry).
    - Multi-light rig (Ambient + Directional + Rim Light + Accent Spot).
    - Smooth idle rotation with mouse-parallax tilt & drag interaction.
    - Built-in WebGL capability detection with animated vector SVG fallback.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PlantCare 3D Hero</title>
        <!-- Pinned Three.js r128 from CDN -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body, html {{
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            #canvas-container {{
                width: 100%;
                height: 100%;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                background: radial-gradient(circle at 50% 50%, rgba(220, 252, 231, 0.6) 0%, rgba(248, 250, 246, 0.2) 65%, rgba(248, 250, 246, 0) 100%);
                border-radius: 18px;
                border: 1px solid rgba(187, 247, 208, 0.6);
            }}
            canvas {{
                width: 100% !important;
                height: 100% !important;
                display: block;
                outline: none;
                cursor: grab;
            }}
            canvas:active {{
                cursor: grabbing;
            }}
            .hud-controls {{
                position: absolute;
                bottom: 12px;
                left: 14px;
                display: flex;
                gap: 8px;
                pointer-events: auto;
            }}
            .hud-btn {{
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(8px);
                border: 1px solid #86EFAC;
                color: #166534;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 14px;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            }}
            .hud-btn:hover {{
                background: #16A34A;
                color: #FFFFFF;
                border-color: #15803D;
            }}
            .badge-overlay {{
                position: absolute;
                top: 12px;
                right: 16px;
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                border: 1px solid rgba(187, 247, 208, 0.8);
                color: #166534;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 12px;
                border-radius: 20px;
                letter-spacing: 0.03em;
                pointer-events: none;
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .pulse-dot {{
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: #22C55E;
                display: inline-block;
                animation: pulseGlow 2s infinite;
            }}
            @keyframes pulseGlow {{
                0%, 100% {{ transform: scale(1); opacity: 1; }}
                50% {{ transform: scale(1.4); opacity: 0.6; }}
            }}
            .fallback-container {{
                display: none;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                width: 100%;
                color: #166534;
                text-align: center;
            }}
            .fallback-leaf {{
                width: 120px;
                height: 120px;
                animation: floatLeaf 4s ease-in-out infinite;
            }}
            @keyframes floatLeaf {{
                0%, 100% {{ transform: translateY(0) rotate(0deg); }}
                50% {{ transform: translateY(-8px) rotate(3deg); }}
            }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <div id="fallback" class="fallback-container">
                <svg class="fallback-leaf" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M50 10C28 24 15 50 25 80C35 90 65 90 75 80C85 50 72 24 50 10Z" fill="#16A34A" fill-opacity="0.85"/>
                    <path d="M50 15V85" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
                    <path d="M50 35L35 48M50 50L30 65M50 35L65 48M50 50L70 65" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <div style="margin-top: 10px; font-weight: 700; font-size: 13px;">Botanical Vision Core</div>
            </div>
            <div class="badge-overlay">
                <span class="pulse-dot"></span>
                <span>Interactive 3D Specimen</span>
            </div>
            <div class="hud-controls">
                <button class="hud-btn" id="btn-toggle-rot">⏸️ Pause</button>
                <button class="hud-btn" id="btn-reset-view">🔄 Reset Angle</button>
            </div>
        </div>

        <script>
            function isWebGLSupported() {{
                try {{
                    var canvas = document.createElement('canvas');
                    return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
                }} catch (e) {{
                    return false;
                }}
            }}

            if (!isWebGLSupported() || typeof THREE === 'undefined') {{
                document.getElementById('fallback').style.display = 'flex';
            }} else {{
                try {{
                    initThreeHero();
                }} catch(err) {{
                    console.error("3D init error:", err);
                    document.getElementById('fallback').style.display = 'flex';
                }}
            }}

            function initThreeHero() {{
                var container = document.getElementById('canvas-container');
                var width = container.clientWidth || 600;
                var height = container.clientHeight || {height};

                // Scene, Camera & Renderer
                var scene = new THREE.Scene();
                var camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
                camera.position.set(0, 0.35, 4.2);

                var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(width, height);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                container.appendChild(renderer.domElement);

                // Lighting Rig
                var ambientLight = new THREE.AmbientLight(0xEBF2E8, 0.95);
                scene.add(ambientLight);

                var dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.2);
                dirLight.position.set(3, 5, 3);
                dirLight.castShadow = true;
                scene.add(dirLight);

                var rimLight = new THREE.DirectionalLight(0x86EFAC, 0.8);
                rimLight.position.set(-3, -2, -2);
                scene.add(rimLight);

                var spotLight = new THREE.PointLight(0x10B981, 1.4, 10);
                spotLight.position.set(0, 2, 2.5);
                scene.add(spotLight);

                // Plant Model Group
                var plantGroup = new THREE.Group();

                // Leaf Geometry Generator
                function createLeafGeometry() {{
                    var shape = new THREE.Shape();
                    shape.moveTo(0, -1.25);
                    shape.bezierCurveTo(0.65, -0.6, 0.95, 0.45, 0, 1.45);
                    shape.bezierCurveTo(-0.95, 0.45, -0.65, -0.6, 0, -1.25);

                    var extrudeSettings = {{
                        steps: 2,
                        depth: 0.045,
                        bevelEnabled: true,
                        bevelThickness: 0.02,
                        bevelSize: 0.02,
                        bevelSegments: 3
                    }};
                    return new THREE.ExtrudeGeometry(shape, extrudeSettings);
                }}

                // Materials
                var leafMatEmerald = new THREE.MeshStandardMaterial({{
                    color: 0x16A34A,
                    roughness: 0.35,
                    metalness: 0.1,
                    flatShading: true,
                    side: THREE.DoubleSide
                }});

                var leafMatSage = new THREE.MeshStandardMaterial({{
                    color: 0x22C55E,
                    roughness: 0.38,
                    metalness: 0.08,
                    flatShading: true,
                    side: THREE.DoubleSide
                }});

                var stemMat = new THREE.MeshStandardMaterial({{
                    color: 0x15803D,
                    roughness: 0.5,
                    metalness: 0.1
                }});

                var veinMat = new THREE.MeshStandardMaterial({{
                    color: 0x86EFAC,
                    roughness: 0.35
                }});

                // Main Stem
                var stemCurve = new THREE.CatmullRomCurve3([
                    new THREE.Vector3(0, -1.6, 0),
                    new THREE.Vector3(0.06, -0.8, 0.04),
                    new THREE.Vector3(-0.03, 0.3, 0),
                    new THREE.Vector3(0, 1.25, -0.04)
                ]);
                var stemGeo = new THREE.TubeGeometry(stemCurve, 24, 0.038, 8, false);
                var stemMesh = new THREE.Mesh(stemGeo, stemMat);
                plantGroup.add(stemMesh);

                // Leaf 1 (Central Primary Leaf)
                var leafGeo1 = createLeafGeometry();
                var leaf1 = new THREE.Mesh(leafGeo1, leafMatEmerald);
                leaf1.scale.set(0.9, 0.9, 0.9);
                leaf1.rotation.set(-0.1, 0, -0.05);
                leaf1.position.set(0, 0.1, 0.02);
                plantGroup.add(leaf1);

                // Central Rib Vein on Leaf 1
                var veinGeo1 = new THREE.CylinderGeometry(0.015, 0.005, 2.3, 6);
                var veinMesh1 = new THREE.Mesh(veinGeo1, veinMat);
                veinMesh1.position.set(0, 0.1, 0.06);
                plantGroup.add(veinMesh1);

                // Leaf 2 (Left Branch Leaf)
                var leafGeo2 = createLeafGeometry();
                var leaf2 = new THREE.Mesh(leafGeo2, leafMatSage);
                leaf2.scale.set(0.68, 0.68, 0.68);
                leaf2.rotation.set(0.2, -0.4, 0.7);
                leaf2.position.set(-0.62, -0.2, 0.15);
                plantGroup.add(leaf2);

                // Leaf 3 (Right Branch Leaf)
                var leafGeo3 = createLeafGeometry();
                var leaf3 = new THREE.Mesh(leafGeo3, leafMatEmerald);
                leaf3.scale.set(0.62, 0.62, 0.62);
                leaf3.rotation.set(-0.15, 0.4, -0.75);
                leaf3.position.set(0.62, -0.4, -0.1);
                plantGroup.add(leaf3);

                // Leaf 4 (Top Shoot)
                var leafGeo4 = createLeafGeometry();
                var leaf4 = new THREE.Mesh(leafGeo4, leafMatSage);
                leaf4.scale.set(0.42, 0.42, 0.42);
                leaf4.rotation.set(0.1, 0.2, 0.1);
                leaf4.position.set(0, 1.25, 0.05);
                plantGroup.add(leaf4);

                scene.add(plantGroup);

                // Particle Spores Constellation
                var particleGeo = new THREE.BufferGeometry();
                var particleCount = 50;
                var posArray = new Float32Array(particleCount * 3);
                for(var i=0; i < particleCount * 3; i+=3) {{
                    var angle = (i / 3) * (Math.PI * 2 / particleCount);
                    var radius = 1.65 + (Math.random() - 0.5) * 0.5;
                    posArray[i] = Math.cos(angle) * radius;
                    posArray[i+1] = (Math.random() - 0.5) * 2.0;
                    posArray[i+2] = Math.sin(angle) * radius;
                }}
                particleGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
                var particleMat = new THREE.PointsMaterial({{
                    size: 0.04,
                    color: 0x86EFAC,
                    transparent: true,
                    opacity: 0.75
                }});
                var particles = new THREE.Points(particleGeo, particleMat);
                scene.add(particles);

                // State & Interactions
                var isAutoRotate = true;
                var isDragging = false;
                var prevMouseX = 0, prevMouseY = 0;
                var mouseParallaxX = 0, mouseParallaxY = 0;

                // Buttons
                var btnRot = document.getElementById('btn-toggle-rot');
                btnRot.addEventListener('click', function() {{
                    isAutoRotate = !isAutoRotate;
                    btnRot.textContent = isAutoRotate ? '⏸️ Pause' : '▶️ Rotate';
                }});

                var btnReset = document.getElementById('btn-reset-view');
                btnReset.addEventListener('click', function() {{
                    plantGroup.rotation.set(0, 0, 0);
                    mouseParallaxX = 0;
                    mouseParallaxY = 0;
                }});

                // Drag to rotate
                renderer.domElement.addEventListener('mousedown', function(e) {{
                    isDragging = true;
                    prevMouseX = e.clientX;
                    prevMouseY = e.clientY;
                }});

                window.addEventListener('mouseup', function() {{ isDragging = false; }});

                window.addEventListener('mousemove', function(e) {{
                    var rect = container.getBoundingClientRect();
                    var x = (e.clientX - rect.left) / rect.width;
                    var y = (e.clientY - rect.top) / rect.height;
                    mouseParallaxX = (x - 0.5) * 2;
                    mouseParallaxY = (y - 0.5) * 2;

                    if (isDragging) {{
                        var deltaX = e.clientX - prevMouseX;
                        var deltaY = e.clientY - prevMouseY;
                        plantGroup.rotation.y += deltaX * 0.01;
                        plantGroup.rotation.x += deltaY * 0.01;
                        prevMouseX = e.clientX;
                        prevMouseY = e.clientY;
                    }}
                }});

                // Window Resize
                window.addEventListener('resize', function() {{
                    var w = container.clientWidth || 600;
                    var h = container.clientHeight || {height};
                    camera.aspect = w / h;
                    camera.updateProjectionMatrix();
                    renderer.setSize(w, h);
                }});

                // Render Loop (60 FPS Performance Target)
                var clock = new THREE.Clock();
                function animate() {{
                    requestAnimationFrame(animate);
                    var elapsedTime = clock.getElapsedTime();

                    if (isAutoRotate && !isDragging) {{
                        plantGroup.rotation.y += 0.009;
                    }}

                    // Subtle breathing float
                    if (!isDragging) {{
                        plantGroup.position.y = Math.sin(elapsedTime * 1.5) * 0.06;
                        plantGroup.rotation.z = Math.cos(elapsedTime * 0.9) * 0.025;
                    }}

                    particles.rotation.y += 0.003;
                    renderer.render(scene, camera);
                }}
                animate();
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# ==============================================================================
# 4. 3D XAI GRAD-CAM LEAF SPOTLIGHT
# ==============================================================================

def render_3d_xai_leaf(confidence: float = 0.9, disease_name: str = "Pathology Focus", height: int = 300):
    """
    Renders an interactive 3D leaf model with simulated Grad-CAM heatmap attention
    projected on the lesion region.
    """
    heat_color = "#DC2626" if confidence > 0.6 else "#F59E0B"
    disease_escaped = html.escape(disease_name)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            body, html {{
                margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: transparent;
                font-family: -apple-system, sans-serif;
            }}
            #xai-3d-box {{
                width: 100%; height: 100%; position: relative; border-radius: 14px;
                background: linear-gradient(145deg, #F8FAF6 0%, #EBF2E8 100%);
                border: 1px solid #BBF7D0;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 4px 16px rgba(22, 163, 74, 0.06);
            }}
            .tag {{
                position: absolute; top: 12px; left: 14px;
                background: rgba(255,255,255,0.95);
                border: 1px solid #86EFAC;
                color: #166534; font-size: 11px; font-weight: 700;
                padding: 4px 10px; border-radius: 8px;
            }}
            .legend {{
                position: absolute; bottom: 10px; left: 12px; right: 12px;
                display: flex; align-items: center; justify-content: space-between;
                font-size: 11px; color: #374151; background: rgba(255,255,255,0.92);
                padding: 6px 12px; border-radius: 8px; border: 1px solid #E5E7EB;
            }}
            .heat-dot {{
                width: 10px; height: 10px; border-radius: 50%; background: {heat_color}; display: inline-block; margin-right: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="xai-3d-box">
            <div class="tag">🔬 3D Surface Attention Projection</div>
            <div class="legend">
                <span><span class="heat-dot"></span>Lesion Focal Zone ({disease_escaped})</span>
                <span style="color: #16A34A; font-weight: 600;">Drag to Inspect</span>
            </div>
        </div>

        <script>
            var container = document.getElementById('xai-3d-box');
            var w = container.clientWidth || 380;
            var h = container.clientHeight || {height};

            var scene = new THREE.Scene();
            var camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
            camera.position.set(0, 0, 3.4);

            var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(w, h);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            scene.add(new THREE.AmbientLight(0xF0FDF4, 1.0));
            var light = new THREE.DirectionalLight(0xFFFFFF, 0.9);
            light.position.set(2, 3, 2);
            scene.add(light);

            var leafGroup = new THREE.Group();

            // Leaf base shape
            var shape = new THREE.Shape();
            shape.moveTo(0, -1.15);
            shape.bezierCurveTo(0.72, -0.5, 0.92, 0.5, 0, 1.35);
            shape.bezierCurveTo(-0.92, 0.5, -0.72, -0.5, 0, -1.15);

            var leafGeo = new THREE.ExtrudeGeometry(shape, {{ depth: 0.035, bevelEnabled: true, bevelThickness: 0.015, bevelSize: 0.015 }});
            var leafMat = new THREE.MeshStandardMaterial({{
                color: 0x15803D,
                roughness: 0.4,
                side: THREE.DoubleSide
            }});
            var leaf = new THREE.Mesh(leafGeo, leafMat);
            leafGroup.add(leaf);

            // Grad-CAM Heatmap Projected Lesion Sphere
            var spotGeo = new THREE.SphereGeometry(0.38, 16, 16);
            spotGeo.scale(1, 0.8, 0.25);
            var spotMat = new THREE.MeshStandardMaterial({{
                color: { "0xDC2626" if confidence > 0.6 else "0xF59E0B" },
                emissive: { "0x991B1B" if confidence > 0.6 else "0xD97706" },
                roughness: 0.3,
                transparent: true,
                opacity: 0.85
            }});
            var heatSpot = new THREE.Mesh(spotGeo, spotMat);
            heatSpot.position.set(0.18, 0.24, 0.04);
            leafGroup.add(heatSpot);

            // Minor Secondary Attention Zone
            var spotGeo2 = new THREE.SphereGeometry(0.22, 12, 12);
            spotGeo2.scale(0.9, 0.9, 0.2);
            var spotMat2 = new THREE.MeshStandardMaterial({{
                color: 0xFBBF24,
                emissive: 0xB45309,
                transparent: true,
                opacity: 0.75
            }});
            var heatSpot2 = new THREE.Mesh(spotGeo2, spotMat2);
            heatSpot2.position.set(-0.16, -0.22, 0.035);
            leafGroup.add(heatSpot2);

            scene.add(leafGroup);

            // Mouse Drag interaction
            var isDragging = false;
            var prevMousePos = {{ x: 0, y: 0 }};

            renderer.domElement.addEventListener('mousedown', function(e) {{
                isDragging = true;
                prevMousePos = {{ x: e.clientX, y: e.clientY }};
            }});

            window.addEventListener('mouseup', function() {{ isDragging = false; }});

            window.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;
                var deltaX = e.clientX - prevMousePos.x;
                var deltaY = e.clientY - prevMousePos.y;
                leafGroup.rotation.y += deltaX * 0.012;
                leafGroup.rotation.x += deltaY * 0.012;
                prevMousePos = {{ x: e.clientX, y: e.clientY }};
            }});

            function animate() {{
                requestAnimationFrame(animate);
                if (!isDragging) {{
                    leafGroup.rotation.y += 0.007;
                }}
                renderer.render(scene, camera);
            }}
            animate();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# ==============================================================================
# 5. DIAGNOSIS CARD & CONFIDENCE METER HELPERS
# ==============================================================================

def get_confidence_band(confidence: float, threshold: float = 0.60) -> tuple[str, str, str]:
    """
    Returns (band_name, badge_css_class, color_hex) for a given confidence score.
    Pairs color with text for high accessibility.
    """
    if confidence < threshold:
        return "Low Confidence / Uncertain", "badge-low", "#DC2626"
    elif confidence >= 0.75:
        return "High Confidence", "badge-high", "#16A34A"
    else:
        return "Moderate Confidence", "badge-mod", "#D97706"


def render_confidence_indicator(confidence: float, threshold: float = 0.60, lang: str = "en"):
    """Renders accessible visual confidence meter with progress bar and explicit text badge."""
    band_name, css_class, color_hex = get_confidence_band(confidence, threshold)
    pct = confidence * 100.0

    st.markdown(
        f"""
        <div style="margin: 14px 0 18px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 0.92rem; color: #14281D;">🎯 {get_text('confidence', lang)}</span>
                <span class="result-badge {css_class}">
                    {band_name} • {pct:.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Streamlit progress bar
    st.progress(min(1.0, max(0.0, float(confidence))))


def render_safety_alert(lang: str = "en"):
    """Renders prominent safety disclaimer when diagnosis confidence falls below safety threshold."""
    st.markdown(
        f"""
        <div class="safety-alert-banner" role="alert">
            <div style="font-size: 1.8rem; line-height: 1;">⚠️</div>
            <div>
                <div class="safety-alert-title">{get_text('uncertain_warning_title', lang)}</div>
                <p class="safety-alert-text">{get_text('uncertain_warning', lang)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 6. HISTORY GALLERY CARD COMPONENT
# ==============================================================================

def render_history_gallery(records: List[Dict[str, Any]], lang: str = "en"):
    """Renders past diagnoses as an accessible, responsive visual gallery."""
    if not records:
        st.info(get_text("history_empty", lang))
        return

    for rec in records:
        plant = html.escape(str(rec.get("plant_name", "Unknown Plant")))
        disease = html.escape(str(rec.get("predicted_disease", "Unknown Disease")))
        conf = float(rec.get("confidence", 0.0))
        model = html.escape(str(rec.get("model_name", "AI Model")))
        timestamp = html.escape(str(rec.get("timestamp", "N/A")))
        is_uncertain = bool(rec.get("flagged_uncertain", 0))

        band_name, css_class, color_hex = get_confidence_band(conf)
        status_badge = f'<span class="result-badge badge-low">{get_text("flagged_uncertain_badge", lang)}</span>' if is_uncertain else f'<span class="result-badge badge-high">{get_text("flagged_confident_badge", lang)}</span>'

        st.markdown(
            f"""
            <div class="history-gallery-card">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="font-size: 2rem; background: #F0FDF4; width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 1px solid #BBF7D0; flex-shrink: 0;">
                        🌱
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #064E3B;">{plant} — {disease}</div>
                        <div style="font-size: 0.82rem; color: #4B6354; margin-top: 2px;">
                            🕒 {timestamp} &nbsp;|&nbsp; 🤖 {model}
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px; flex-shrink: 0;">
                    <span class="result-badge {css_class}">{conf * 100:.1f}%</span>
                    {status_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
