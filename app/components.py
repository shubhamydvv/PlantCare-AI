"""PlantCare AI - Production-grade Calm Botanical UI Components, Stepper, 3D Visualizer & Localization."""

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
        "system_live": "System Live • 100K Catalog Active",
        "app_badge": "AI BOTANICAL DIAGNOSTICS",
        "title": "PlantCare AI",
        "title_full": "🌿 PlantCare AI — Explainable Plant Disease Diagnosis",
        "subtitle": "Clinical-grade crop pathology detection with deep neural networks & visual explainability (XAI)",
        "hero_value_prop": "AI-Powered Explainable Plant Disease Diagnosis & Treatment",
        "hero_how_it_works": "1. Upload Photo → 2. Deep AI Diagnosis → 3. Grad-CAM Explanation → 4. Agronomic Guidance",
        "step_1": "1. Upload Photo",
        "step_2": "2. AI Analysis",
        "step_3": "3. XAI Heatmap",
        "step_4": "4. Treatment Plan",
        "step_1_desc": "Select or capture leaf photo",
        "step_2_desc": "Neural pathology inference",
        "step_3_desc": "Grad-CAM lesion attention",
        "step_4_desc": "Extension-backed guidance",
        "disclaimer": "⚠️ Academic Research Prototype (Chandigarh University) • Recommendation guidance is sourced from FAO, ICAR, Cornell & UC Davis. Consult an agricultural extension specialist before applying heavy chemical treatments.",
        "sidebar_title": "Configuration",
        "select_model": "Neural Architecture",
        "confidence_threshold": "Safety Cutoff Threshold (%)",
        "confidence_threshold_help": "Predictions below this threshold are flagged as uncertain to prevent misapplication of chemical treatments.",
        "select_plant": "Select Crop / Plant Species",
        "upload_label": "Upload Leaf Specimen",
        "upload_help": "Supported formats: JPG, JPEG, PNG, WebP (Max 10MB)",
        "upload_drag_hint": "Drop a leaf photo here or click to browse",
        "camera_label": "Or Capture via Camera",
        "diagnose_btn": "⚡ Run AI Diagnosis",
        "analyzing_progress": "Scanning foliar tissue & computing Grad-CAM saliency...",
        "preview_title": "Leaf Specimen Preview",
        "preview_ready": "Specimen loaded. Click 'Run AI Diagnosis' to analyze.",
        "results_title": "Diagnostic Report",
        "plant_species": "Plant Species",
        "prediction": "Identified Pathology",
        "confidence": "Diagnosis Confidence",
        "confidence_band_high": "High Confidence",
        "confidence_band_moderate": "Moderate Confidence",
        "confidence_band_low": "Uncertain — Verification Required",
        "uncertain_warning_title": "⚠️ Safety Alert: Low Confidence Diagnosis",
        "uncertain_warning": "Prediction confidence is below the safety cutoff. Do not apply intensive chemical treatments without specialist verification.",
        "tab_xai": "🔬 XAI Saliency",
        "tab_symptoms": "🔍 Symptoms",
        "tab_prevention": "🛡️ Prevention",
        "tab_treatment": "💊 Treatment",
        "tab_citations": "📚 Source Citation",
        "tab_3d_view": "🌿 3D Saliency Model",
        "xai_title": "Visual Attention & Saliency Map (Grad-CAM)",
        "xai_desc": "Warm colors (red/orange/yellow) indicate the exact leaf regions that influenced the model's diagnostic prediction.",
        "orig_image": "Original Input Leaf",
        "xai_overlay": "Grad-CAM Saliency Overlay",
        "symptoms": "Clinical Pathology Symptoms",
        "prevention": "Cultural & Preventative Protocol",
        "treatment": "Therapeutic Treatment Guidance",
        "source": "Verified Agricultural Extension Citation",
        "recent_history": "Recent Diagnostic Telemetry",
        "history_empty": "No leaf diagnoses recorded yet. Upload a leaf image above to start.",
        "model_comparison": "Benchmark Metrics & Robustness",
        "clear_history": "Clear History",
        "three_d_hero_caption": "Interactive 3D Botanical Specimen",
        "three_d_xai_caption": "3D Interactive Leaf Specimen with Projected Attention Field",
        "webgl_fallback": "Botanical Vision Core",
        "accuracy_f1_chart": "F1-Score Comparison under Clean vs Degraded Robustness Conditions",
        "latency_chart": "Inference Latency vs Macro F1 Trade-off",
        "calibration_chart": "Confidence Reliability Diagrams",
        "view_details": "View Details",
        "flagged_uncertain_badge": "Safety Flagged",
        "flagged_confident_badge": "Verified",
        "sample_tray_title": "⚡ Quick Test Specimens",
        "sample_tray_hint": "Select any preloaded leaf specimen to test the diagnostic pipeline instantly",
        "top_predictions_title": "Probable Pathogen Distribution",
        "download_report_btn": "📥 Export Diagnostic Report (TXT)",
    },
    "hi": {
        "system_live": "सिस्टम लाइव • 100K कैटलॉग सक्रिय",
        "app_badge": "एआई पादप निदान",
        "title": "प्लांटकेयर एआई",
        "title_full": "🌿 PlantCare AI — व्याख्यात्मक पादप रोग निदान",
        "subtitle": "डीप न्यूरल नेटवर्क और व्याख्यात्मक एआई (XAI) द्वारा संचालित सटीक पादप विकृति विज्ञान",
        "hero_value_prop": "स्वस्थ फसलों के लिए एआई-संचालित व्याख्यात्मक पादप रोग निदान",
        "hero_how_it_works": "1. फोटो अपलोड → 2. एआई विश्लेषण → 3. XAI हीटमैप → 4. उपचार योजना",
        "step_1": "1. फोटो अपलोड",
        "step_2": "2. एआई विश्लेषण",
        "step_3": "3. XAI हीटमैप",
        "step_4": "4. उपचार योजना",
        "step_1_desc": "पत्ती की तस्वीर चुनें या खींचें",
        "step_2_desc": "न्यूरल नेटवर्क द्वारा स्कैन",
        "step_3_desc": "Grad-CAM ध्यान स्पष्टीकरण",
        "step_4_desc": "सत्यापित उपचार मार्गदर्शन",
        "disclaimer": "⚠️ अकादमिक अनुसंधान प्रोटोटाइप (चंडीगढ़ विश्वविद्यालय) • मार्गदर्शन FAO, ICAR, Cornell एवं UC Davis से लिया गया है। रासायनिक उपचार से पहले कृषि विशेषज्ञ से परामर्श लें।",
        "sidebar_title": "कॉन्फ़िगरेशन",
        "select_model": "न्यूरल आर्किटेक्चर चुनें",
        "confidence_threshold": "सुरक्षा सीमा (%)",
        "confidence_threshold_help": "इस सीमा से कम विश्वास वाले निदानों को अनिश्चित चिह्नित किया जाता है ताकि गलत उपचार से बचा जा सके।",
        "select_plant": "फसल / पौधे की प्रजाति चुनें",
        "upload_label": "पत्ती की तस्वीर अपलोड करें",
        "upload_help": "स्वीकृत प्रारूप: JPG, JPEG, PNG, WebP (अधिकतम 10MB)",
        "upload_drag_hint": "पत्ती की छवि चुनने के लिए क्लिक करें",
        "camera_label": "या कैमरे से तस्वीर लें",
        "diagnose_btn": "⚡ एआई जांच शुरू करें",
        "analyzing_progress": "पत्ती के ऊतकों का विश्लेषण एवं Grad-CAM ध्यान गणना जारी...",
        "preview_title": "चयनित नमूने का पूर्वावलोकन",
        "preview_ready": "नमूना लोड हो गया। जांच शुरू करने के लिए क्लिक करें।",
        "results_title": "निदान रिपोर्ट",
        "plant_species": "पौधे की प्रजाति",
        "prediction": "पहचाना गया रोग",
        "confidence": "निदान विश्वास",
        "confidence_band_high": "उच्च विश्वास (High)",
        "confidence_band_moderate": "मध्यम विश्वास (Moderate)",
        "confidence_band_low": "अनिश्चित — विशेषज्ञ सत्यापन आवश्यक",
        "uncertain_warning_title": "⚠️ कम विश्वास निदान चेतावनी",
        "uncertain_warning": "निदान का विश्वास सुरक्षा सीमा से कम है। किसी विशेषज्ञ से परामर्श किए बिना गहन रासायनिक उपचार न करें।",
        "tab_xai": "🔬 XAI स्पष्टीकरण",
        "tab_symptoms": "🔍 लक्षण",
        "tab_prevention": "🛡️ रोकथाम",
        "tab_treatment": "💊 उपचार",
        "tab_citations": "📚 स्रोत",
        "tab_3d_view": "🌿 3D मॉडल",
        "xai_title": "दृश्य ध्यान एवं स्पष्टीकरण (Grad-CAM)",
        "xai_desc": "उजागर किए गए क्षेत्र (लाल/पीले रंग) उन प्रमुख पत्ती के हिस्सों को दर्शाते हैं जिन्होंने मॉडल के निर्णय को निर्देशित किया।",
        "orig_image": "मूल इनपुट तस्वीर",
        "xai_overlay": "Grad-CAM ध्यान ओवरले",
        "symptoms": "रोग के लक्षण",
        "prevention": "रोकथाम एवं सांस्कृतिक उपाय",
        "treatment": "कृषि एवं उपचार दिशानिर्देश",
        "source": "सत्यापित कृषि विस्तार स्रोत",
        "recent_history": "हालिया निदान इतिहास",
        "history_empty": "अभी तक कोई निदान दर्ज नहीं हुआ है।",
        "model_comparison": "मॉडल तुलना और मेट्रिक्स",
        "clear_history": "इतिहास साफ़ करें",
        "three_d_hero_caption": "इंटरैक्टिव 3D वनस्पति मॉडल",
        "three_d_xai_caption": "3D पत्ती नमूना एवं ध्यान क्षेत्र",
        "webgl_fallback": "वानस्पतिक दृष्टि कोर",
        "accuracy_f1_chart": "F1-स्कोर तुलना",
        "latency_chart": "विलंबता बनाम संतुलन",
        "calibration_chart": "विश्वसनीयता आरेख",
        "view_details": "विवरण देखें",
        "flagged_uncertain_badge": "सुरक्षा ध्वज",
        "flagged_confident_badge": "सत्यापित",
        "sample_tray_title": "⚡ त्वरित परीक्षण नमूने",
        "sample_tray_hint": "तुरंत परीक्षण के लिए किसी भी नमूने पर क्लिक करें",
        "top_predictions_title": "संभावित रोग वितरण",
        "download_report_btn": "📥 रिपोर्ट डाउनलोड करें",
    },
}


def get_text(key: str, lang: str = "en") -> str:
    """Returns localized string with fallback to English."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ==============================================================================
# 2. CALM BOTANICAL DESIGN SYSTEM & RESPONSIVE CSS INJECTION
# ==============================================================================

def inject_custom_styles():
    """Injects cohesive calm botanical styling with strict 8px spacing, clean cards, and responsive layouts."""
    st.markdown(
        """
        <style>
        /* Import Clean Typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            /* Calm Botanical Palette */
            --botanical-bg: #F8FAF7;
            --botanical-surface: #FFFFFF;
            --botanical-surface-glass: rgba(255, 255, 255, 0.92);
            --botanical-border: #E5E9E2;
            --botanical-border-light: #F0F4ED;
            
            --botanical-primary-900: #064E3B; /* Deep Forest */
            --botanical-primary-800: #065F46;
            --botanical-primary-700: #047857;
            --botanical-primary-600: #059669;
            --botanical-primary-500: #10B981; /* Leaf Green */
            --botanical-primary-100: #DCFCE7;
            --botanical-primary-50:  #F0FDF4;

            --botanical-text-main: #132A1C;
            --botanical-text-secondary: #4A5E51;
            --botanical-text-muted: #7D9285;

            --botanical-amber: #D97706;
            --botanical-amber-light: #FEF3C7;
            --botanical-red: #DC2626;
            --botanical-red-light: #FEE2E2;

            /* Strict 8px Spacing Tokens */
            --space-1: 8px;
            --space-2: 16px;
            --space-3: 24px;
            --space-4: 32px;
            --space-5: 40px;

            /* Radii */
            --radius-sm: 10px;
            --radius-md: 16px;
            --radius-lg: 22px;
            --radius-full: 9999px;

            /* Shadows */
            --shadow-sm: 0 2px 6px rgba(6, 78, 59, 0.04);
            --shadow-card: 0 4px 18px -2px rgba(6, 78, 59, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
            --shadow-hover: 0 10px 26px -4px rgba(6, 78, 59, 0.10), 0 4px 10px -2px rgba(0, 0, 0, 0.03);
        }

        /* Base Canvas & Typography */
        .stApp {
            background-color: var(--botanical-bg);
            background-image: 
                radial-gradient(at 10% 10%, rgba(16, 185, 129, 0.05) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(6, 78, 59, 0.04) 0px, transparent 50%);
            background-attachment: fixed;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
            color: var(--botanical-text-main);
        }

        h1, h2, h3, h4, h5, h6, .stTitle {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: var(--botanical-text-main) !important;
        }

        /* ----------------------------------------------------------------------
           COMPACT HERO BANNER
        ---------------------------------------------------------------------- */
        .compact-hero-card {
            background: linear-gradient(135deg, #064E3B 0%, #065F46 65%, #047857 100%);
            border-radius: var(--radius-lg);
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px -4px rgba(6, 78, 59, 0.25);
            color: #FFFFFF;
            position: relative;
            overflow: hidden;
        }

        .compact-hero-card::after {
            content: "";
            position: absolute;
            top: -40px;
            right: -40px;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.3) 0%, transparent 70%);
            pointer-events: none;
        }

        .hero-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 12px;
        }

        .hero-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.16);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #ECFDF5;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 5px 14px;
            border-radius: var(--radius-full);
        }

        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #34D399;
            box-shadow: 0 0 10px #34D399;
            animation: livePulse 2s infinite;
        }

        @keyframes livePulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .hero-title {
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.2;
            color: #FFFFFF !important;
            margin-bottom: 6px;
        }

        .hero-desc {
            font-size: 0.95rem;
            color: #D1FAE5;
            line-height: 1.5;
            max-width: 680px;
            margin-bottom: 16px;
        }

        .hero-metrics-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .hero-metric-chip {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 6px 14px;
            border-radius: var(--radius-full);
            font-size: 0.8rem;
            font-weight: 600;
            color: #F0FDF4;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        /* ----------------------------------------------------------------------
           HORIZONTAL WORKFLOW STEPPER
        ---------------------------------------------------------------------- */
        .stepper-container {
            background: var(--botanical-surface);
            border: 1px solid var(--botanical-border);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-card);
        }

        .stepper-track {
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            gap: 8px;
        }

        .stepper-item {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
            min-width: 130px;
        }

        .stepper-badge {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            background: var(--botanical-primary-50);
            border: 1.5px solid var(--botanical-primary-500);
            color: var(--botanical-primary-800);
            flex-shrink: 0;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.15);
        }

        .stepper-text-title {
            font-size: 0.84rem;
            font-weight: 700;
            color: var(--botanical-text-main);
            line-height: 1.2;
        }

        .stepper-text-desc {
            font-size: 0.74rem;
            color: var(--botanical-text-muted);
        }

        .stepper-arrow {
            color: var(--botanical-border);
            font-size: 1.1rem;
            padding: 0 4px;
        }

        /* ----------------------------------------------------------------------
           SPECIMEN QUICK-TEST CARDS
        ---------------------------------------------------------------------- */
        .specimen-card-box {
            background: var(--botanical-surface);
            border: 1px solid var(--botanical-border);
            border-radius: var(--radius-md);
            padding: 18px 20px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-card);
        }

        .specimen-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }

        /* ----------------------------------------------------------------------
           BOTANICAL CARDS & SURFACES
        ---------------------------------------------------------------------- */
        .botanical-card {
            background: var(--botanical-surface);
            border: 1px solid var(--botanical-border);
            border-radius: var(--radius-md);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: var(--shadow-card);
            transition: all 0.2s ease;
        }

        .botanical-card:hover {
            box-shadow: var(--shadow-hover);
        }

        /* Streamlit Buttons Styled Cleanly */
        .stButton > button {
            background: #FFFFFF !important;
            color: var(--botanical-text-main) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            border: 1px solid var(--botanical-border) !important;
            border-radius: var(--radius-full) !important;
            padding: 8px 18px !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stButton > button:hover {
            background: var(--botanical-primary-50) !important;
            border-color: var(--botanical-primary-500) !important;
            color: var(--botanical-primary-800) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* Primary Action Button (Deep Leaf Green) */
        div[data-testid="stVerticalBlock"] > div > .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 6px 18px rgba(4, 120, 87, 0.3) !important;
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 8px 24px rgba(4, 120, 87, 0.4) !important;
            color: #FFFFFF !important;
            transform: translateY(-2px);
        }

        /* Accessible Confidence Badges */
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: var(--radius-full);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .badge-high {
            background: var(--botanical-primary-100);
            color: #15803D;
            border: 1px solid #86EFAC;
        }

        .badge-mod {
            background: var(--botanical-amber-light);
            color: #B45309;
            border: 1px solid #FCD34D;
        }

        .badge-low {
            background: var(--botanical-red-light);
            color: #B91C1C;
            border: 1px solid #FCA5A5;
        }

        /* Modern Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #EBF2E8;
            padding: 6px;
            border-radius: var(--radius-full);
            margin-bottom: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-full) !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: var(--botanical-text-secondary) !important;
            border: none !important;
            background: transparent !important;
            transition: all 0.2s ease !important;
        }

        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: var(--botanical-primary-900) !important;
            box-shadow: 0 2px 8px rgba(6, 78, 59, 0.08) !important;
        }

        /* History Cards */
        .history-gallery-card {
            background: var(--botanical-surface);
            border: 1px solid var(--botanical-border);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }

        .history-gallery-card:hover {
            border-color: var(--botanical-primary-500);
            transform: translateX(4px);
        }

        /* Subtle Footer Disclaimer */
        .subtle-footer-disclaimer {
            background: #F1F5F0;
            border: 1px solid #E2E8DF;
            border-radius: var(--radius-md);
            padding: 14px 20px;
            color: #4A5E51;
            font-size: 0.82rem;
            line-height: 1.5;
            text-align: center;
            margin-top: 36px;
            margin-bottom: 16px;
        }

        /* Low Confidence Alert Banner */
        .safety-alert-banner {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-radius: var(--radius-md);
            padding: 16px 20px;
            color: #92400E;
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
        }

        /* Responsive Breakpoints */
        @media (max-width: 768px) {
            .stepper-track {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            .stepper-arrow {
                display: none;
            }
            .compact-hero-card {
                padding: 20px 18px;
            }
            .hero-title {
                font-size: 1.5rem;
            }
            .hero-metrics-bar {
                flex-direction: column;
                gap: 6px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. COMPACT HERO BANNER & HORIZONTAL STEPPER COMPONENTS
# ==============================================================================

def render_compact_hero(
    model_name: str = "resnet50",
    confidence_threshold: float = 0.60,
    device: str = "cpu",
    lang: str = "en",
):
    """Renders high-impact, compact hero and live operational telemetry banner."""
    model_display_names = {
        "resnet50": "ResNet50 (Deep CNN)",
        "efficientnet_b0": "EfficientNet-B0",
        "vit_b_16": "ViT-B/16 (Transformer)",
    }
    m_name = model_display_names.get(model_name, model_name)
    threshold_pct = int(confidence_threshold * 100)

    st.markdown(
        f"""
        <div class="compact-hero-card">
            <div class="hero-top-row">
                <div class="hero-status-pill">
                    <span class="live-dot"></span>
                    <span>{get_text('system_live', lang)}</span>
                </div>
                <div style="font-size: 0.8rem; color: #A7F3D0; font-weight: 600;">
                    {get_text('app_badge', lang)} v2.4
                </div>
            </div>
            <div class="hero-title">{get_text('title', lang)} — {get_text('hero_value_prop', lang)}</div>
            <div class="hero-desc">{get_text('subtitle', lang)}</div>
            <div class="hero-metrics-bar">
                <div class="hero-metric-chip">🧠 <strong>Model:</strong> {m_name}</div>
                <div class="hero-metric-chip">🛡️ <strong>Safety Cutoff:</strong> {threshold_pct}%</div>
                <div class="hero-metric-chip">⚡ <strong>Compute:</strong> {device.upper()}</div>
                <div class="hero-metric-chip">🗄️ <strong>Dataset:</strong> 100,000 Cataloged</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_stepper(current_step: int = 1, lang: str = "en"):
    """Renders a responsive 4-step horizontal progress tracker."""
    st.markdown(
        f"""
        <div class="stepper-container">
            <div class="stepper-track">
                <div class="stepper-item">
                    <div class="stepper-badge">1</div>
                    <div>
                        <div class="stepper-text-title">{get_text('step_1', lang)}</div>
                        <div class="stepper-text-desc">{get_text('step_1_desc', lang)}</div>
                    </div>
                </div>
                <div class="stepper-arrow">➔</div>
                <div class="stepper-item">
                    <div class="stepper-badge">2</div>
                    <div>
                        <div class="stepper-text-title">{get_text('step_2', lang)}</div>
                        <div class="stepper-text-desc">{get_text('step_2_desc', lang)}</div>
                    </div>
                </div>
                <div class="stepper-arrow">➔</div>
                <div class="stepper-item">
                    <div class="stepper-badge">3</div>
                    <div>
                        <div class="stepper-text-title">{get_text('step_3', lang)}</div>
                        <div class="stepper-text-desc">{get_text('step_3_desc', lang)}</div>
                    </div>
                </div>
                <div class="stepper-arrow">➔</div>
                <div class="stepper-item">
                    <div class="stepper-badge">4</div>
                    <div>
                        <div class="stepper-text-title">{get_text('step_4', lang)}</div>
                        <div class="stepper-text-desc">{get_text('step_4_desc', lang)}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_disclaimer(lang: str = "en"):
    """Renders subtle, non-intrusive research disclaimer banner at bottom of page."""
    st.markdown(
        f"""
        <div class="subtle-footer-disclaimer">
            {get_text('disclaimer', lang)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 4. 3D THREE.JS PROCEDURAL SPECIMEN & XAI VISUALIZERS
# ==============================================================================

def render_3d_plant_hero(height: int = 300):
    """
    Renders an interactive, procedural 3D botanical specimen using Three.js inside an iframe.
    - Zero external 3D asset downloads (pure procedural geometry).
    - Multi-light rig (Ambient + Directional + Rim Light).
    - Smooth idle rotation with mouse-parallax tilt & drag interaction.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body, html {{
                width: 100%; height: 100%; overflow: hidden; background: transparent;
                font-family: -apple-system, sans-serif;
            }}
            #canvas-wrap {{
                width: 100%; height: 100%; position: relative;
                display: flex; align-items: center; justify-content: center;
                background: radial-gradient(circle at 50% 50%, rgba(220, 252, 231, 0.4) 0%, rgba(248, 250, 247, 0.1) 65%, transparent 100%);
                border-radius: 18px; border: 1px solid rgba(229, 233, 226, 0.8);
            }}
            canvas {{ width: 100% !important; height: 100% !important; display: block; outline: none; cursor: grab; }}
            canvas:active {{ cursor: grabbing; }}
            .pill-tag {{
                position: absolute; top: 10px; right: 12px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #A7F3D0; color: #065F46; font-size: 11px; font-weight: 700;
                padding: 4px 10px; border-radius: 9999px; pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div id="canvas-wrap">
            <div class="pill-tag">🌿 3D Botanical Model</div>
        </div>
        <script>
            var container = document.getElementById('canvas-wrap');
            var w = container.clientWidth || 500;
            var h = container.clientHeight || {height};

            var scene = new THREE.Scene();
            var camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 1000);
            camera.position.set(0, 0.2, 4.2);

            var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(w, h);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            scene.add(new THREE.AmbientLight(0xF0FDF4, 1.2));
            var dirLight = new THREE.DirectionalLight(0xFFFFFF, 1.2);
            dirLight.position.set(3, 4, 3);
            scene.add(dirLight);

            var plantGroup = new THREE.Group();

            function createLeaf() {{
                var shape = new THREE.Shape();
                shape.moveTo(0, -1.2);
                shape.bezierCurveTo(0.7, -0.6, 0.9, 0.4, 0, 1.4);
                shape.bezierCurveTo(-0.9, 0.4, -0.7, -0.6, 0, -1.2);
                return new THREE.ExtrudeGeometry(shape, {{ steps: 2, depth: 0.035, bevelEnabled: true, bevelThickness: 0.015, bevelSize: 0.015 }});
            }}

            var matGreen = new THREE.MeshStandardMaterial({{ color: 0x059669, roughness: 0.3, side: THREE.DoubleSide }});
            var leaf1 = new THREE.Mesh(createLeaf(), matGreen);
            plantGroup.add(leaf1);

            var leaf2 = new THREE.Mesh(createLeaf(), matGreen);
            leaf2.scale.set(0.65, 0.65, 0.65);
            leaf2.rotation.set(0.2, -0.3, 0.6);
            leaf2.position.set(-0.55, -0.2, 0.1);
            plantGroup.add(leaf2);

            var leaf3 = new THREE.Mesh(createLeaf(), matGreen);
            leaf3.scale.set(0.6, 0.6, 0.6);
            leaf3.rotation.set(-0.15, 0.3, -0.65);
            leaf3.position.set(0.55, -0.3, -0.1);
            plantGroup.add(leaf3);

            scene.add(plantGroup);

            var isDrag = false, lx = 0, ly = 0;
            renderer.domElement.addEventListener('mousedown', function(e) {{ isDrag = true; lx = e.clientX; ly = e.clientY; }});
            window.addEventListener('mouseup', function() {{ isDrag = false; }});
            window.addEventListener('mousemove', function(e) {{
                if(isDrag) {{
                    var dx = e.clientX - lx; var dy = e.clientY - ly;
                    plantGroup.rotation.y += dx * 0.01;
                    plantGroup.rotation.x += dy * 0.01;
                    lx = e.clientX; ly = e.clientY;
                }}
            }});

            function render() {{
                requestAnimationFrame(render);
                if(!isDrag) {{ plantGroup.rotation.y += 0.008; }}
                renderer.render(scene, camera);
            }}
            render();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


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
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body, html {{ width: 100%; height: 100%; overflow: hidden; background: transparent; font-family: -apple-system, sans-serif; }}
            #xai-box {{
                width: 100%; height: 100%; position: relative; border-radius: 16px;
                background: linear-gradient(145deg, #F8FAF7 0%, #EBF2E8 100%);
                border: 1px solid #D1FAE5; display: flex; align-items: center; justify-content: center;
                box-shadow: 0 4px 16px rgba(6, 78, 59, 0.05);
            }}
            .tag {{
                position: absolute; top: 12px; left: 14px;
                background: rgba(255,255,255,0.92); border: 1px solid #A7F3D0;
                color: #065F46; font-size: 11px; font-weight: 700;
                padding: 4px 10px; border-radius: 9999px;
            }}
            .legend {{
                position: absolute; bottom: 10px; left: 12px; right: 12px;
                display: flex; align-items: center; justify-content: space-between;
                font-size: 11px; color: #374151; background: rgba(255,255,255,0.92);
                padding: 6px 12px; border-radius: 9999px; border: 1px solid #E5E7EB;
            }}
            .heat-dot {{
                width: 8px; height: 8px; border-radius: 50%; background: {heat_color}; display: inline-block; margin-right: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="xai-box">
            <div class="tag">🔬 3D Surface Attention Projection</div>
            <div class="legend">
                <span><span class="heat-dot"></span>Lesion Focal Zone ({disease_escaped})</span>
                <span style="color: #059669; font-weight: 600;">Drag to Rotate</span>
            </div>
        </div>
        <script>
            var container = document.getElementById('xai-box');
            var w = container.clientWidth || 380;
            var h = container.clientHeight || {height};

            var scene = new THREE.Scene();
            var camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
            camera.position.set(0, 0, 3.4);

            var renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(w, h);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            scene.add(new THREE.AmbientLight(0xF0FDF4, 1.2));
            var light = new THREE.DirectionalLight(0xFFFFFF, 1.0);
            light.position.set(2, 3, 2);
            scene.add(light);

            var leafGroup = new THREE.Group();

            var shape = new THREE.Shape();
            shape.moveTo(0, -1.15);
            shape.bezierCurveTo(0.72, -0.5, 0.92, 0.5, 0, 1.35);
            shape.bezierCurveTo(-0.92, 0.5, -0.72, -0.5, 0, -1.15);

            var leafGeo = new THREE.ExtrudeGeometry(shape, {{ depth: 0.035, bevelEnabled: true, bevelThickness: 0.015, bevelSize: 0.015 }});
            var leafMat = new THREE.MeshStandardMaterial({{ color: 0x059669, roughness: 0.35, side: THREE.DoubleSide }});
            var leaf = new THREE.Mesh(leafGeo, leafMat);
            leafGroup.add(leaf);

            var spotGeo = new THREE.SphereGeometry(0.38, 16, 16);
            spotGeo.scale(1, 0.8, 0.25);
            var spotMat = new THREE.MeshStandardMaterial({{
                color: { "0xDC2626" if confidence > 0.6 else "0xF59E0B" },
                emissive: { "0x991B1B" if confidence > 0.6 else "0xD97706" },
                roughness: 0.3, transparent: true, opacity: 0.85
            }});
            var heatSpot = new THREE.Mesh(spotGeo, spotMat);
            heatSpot.position.set(0.18, 0.24, 0.04);
            leafGroup.add(heatSpot);

            scene.add(leafGroup);

            var isDrag = false, lx = 0, ly = 0;
            renderer.domElement.addEventListener('mousedown', function(e) {{ isDrag = true; lx = e.clientX; ly = e.clientY; }});
            window.addEventListener('mouseup', function() {{ isDrag = false; }});
            window.addEventListener('mousemove', function(e) {{
                if (isDrag) {{
                    var dx = e.clientX - lx; var dy = e.clientY - ly;
                    leafGroup.rotation.y += dx * 0.012;
                    leafGroup.rotation.x += dy * 0.012;
                    lx = e.clientX; ly = e.clientY;
                }}
            }});

            function animate() {{
                requestAnimationFrame(animate);
                if (!isDrag) {{ leafGroup.rotation.y += 0.007; }}
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
    """Returns (band_name, badge_css_class, color_hex) for a given confidence score."""
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
        <div style="margin: 12px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 0.92rem; color: #132A1C;">🎯 {get_text('confidence', lang)}</span>
                <span class="result-badge {css_class}">
                    {band_name} • {pct:.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(1.0, max(0.0, float(confidence))))


def render_safety_alert(lang: str = "en"):
    """Renders prominent safety disclaimer when diagnosis confidence falls below safety threshold."""
    st.markdown(
        f"""
        <div class="safety-alert-banner" role="alert">
            <div style="font-size: 1.6rem; line-height: 1;">⚠️</div>
            <div>
                <div style="font-weight: 700; font-size: 0.95rem;">{get_text('uncertain_warning_title', lang)}</div>
                <p style="font-size: 0.85rem; margin-top: 2px;">{get_text('uncertain_warning', lang)}</p>
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
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="font-size: 1.6rem; background: #F0FDF4; width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; border: 1px solid #BBF7D0; flex-shrink: 0;">
                        🌱
                    </div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.98rem; color: #132A1C;">{plant} — {disease}</div>
                        <div style="font-size: 0.8rem; color: #7D9285; margin-top: 2px;">
                            🕒 {timestamp} &nbsp;|&nbsp; 🤖 {model}
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; flex-shrink: 0;">
                    <span class="result-badge {css_class}">{conf * 100:.1f}%</span>
                    {status_badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
