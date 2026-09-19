# 🌿 PlantCare AI — Explainable Plant Disease Diagnosis & Treatment Recommendation System

> **A final-year B.E. (CSE – Big Data & Analytics) research project — Chandigarh University**  
> *Developed by Alisha Gupta (23BDA70030) & Shubham (23BDA70087) under the supervision of Mrs. Somdatta Patra.*

---

## 📌 Abstract & Overview

**PlantCare AI** is an end-to-end Explainable AI (XAI) system and web application that diagnoses plant diseases from leaf photographs and provides evidence-based prevention and management recommendations. The system compares three transfer-learning architectures (**ResNet50**, **EfficientNet-B0**, and **Vision Transformer ViT-B/16**) across:
1. **Classification Accuracy & Macro F1-Score**
2. **Robustness & Degradation Resistance** (Brightness shifts, Gaussian blur, Rotations, Additive noise)
3. **Model Confidence Calibration** (Expected Calibration Error & Brier Score)
4. **Visual Interpretability** (Grad-CAM for CNNs, Attention Rollout for ViTs)
5. **Evidence-Based Knowledge Retrieval** (Curated, citable agricultural extension recommendations from ICAR, FAO, and university extensions)

---

## 🏗️ Repository Architecture

```
PlantCare-AI/
├── data/
│   ├── raw/                  # Raw PlantVillage leaf photographs
│   ├── processed/            # Stratified 70/15/15 splits (split_info.json)
│   └── robustness_test_set/  # Persisted degraded test variants (brightness, blur, rotation, noise)
├── models/
│   ├── resnet50/             # Checkpoints & weights for ResNet50
│   ├── efficientnet/         # Checkpoints & weights for EfficientNet-B0
│   └── vit/                  # Checkpoints & weights for ViT-B/16
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # System paths, device detection & label registry
│   ├── dataset.py            # Dataset loader, KaggleHub fetcher & synthetic generator
│   ├── preprocessing.py      # Stratified 70/15/15 split & augmentation transforms
│   ├── robustness.py         # 9 degradation transforms & test benchmark generator
│   ├── models.py             # ResNet50, EfficientNet-B0, ViT-B/16, CNN-ViT Hybrid
│   ├── train.py              # Unified multi-model training engine with run logging
│   ├── evaluate.py           # Evaluation, robustness benchmark & chart generator
│   ├── explainability.py     # Grad-CAM & Attention Rollout implementations
│   ├── recommendations.py    # SQLite knowledge base query & diagnosis history logger
│   └── utils.py              # Image validation, hashing & ONNX exporter
├── database/
│   ├── schema.sql            # SQLite schema
│   ├── seed_data.py          # Extension-sourced knowledge base seed script
│   └── plantcare.db          # Persisted SQLite database
├── app/
│   ├── app.py                # Interactive Streamlit Web Application
│   └── components.py         # UI cards & English/Hindi bilingual strings
├── results/
│   ├── metrics/              # Run logs (run_log.json), CSV comparison tables
│   ├── graphs/               # Publication charts & reliability diagrams
│   └── xai_outputs/          # Saved Grad-CAM and Attention Rollout sample overlays
├── tests/
│   ├── test_preprocessing.py # Stratified split & transform tests
│   ├── test_robustness.py    # Degradation transform unit tests
│   ├── test_models.py        # Forward pass & parameter count tests
│   ├── test_recommendations.py# DB retrieval, citations & fallback tests
│   └── test_app_logic.py     # Upload validation & confidence threshold tests
├── configs/
│   └── train_config.yaml     # Hyperparameters & experiment configuration
├── .github/workflows/
│   └── ci.yml                # Automated CI pipeline
├── Dockerfile                # Docker container build definition
├── requirements.txt          # Pinned exact Python dependencies
└── README.md
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/shubhamydvv/PlantCare-AI.git
cd PlantCare-AI

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Recommendation Knowledge Base
```bash
python -m database.seed_data
```

### 3. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 📊 Dataset Ingestion & Pipeline

### Automated Dataset Fetching
The pipeline supports automated programmatic fetching via `kagglehub`:
```bash
python -m src.preprocessing
```
*Note: If offline or running without Kaggle credentials, `src/dataset.py` automatically initializes a synthetic mock dataset across all 38 classes for immediate testing.*

### Robustness Benchmark Generation
To generate the 9 fixed degradation test variants (Brightness Low/High, Blur Mild/Heavy, Rotation 45/90/180°, Noise Mild/Heavy):
```bash
python -m src.robustness
```

---

## 🚀 Model Training & Evaluation

### Train Architectures
You can train any of the three architectures with one command:
```bash
# 1. Train ResNet50
python -m src.train --model resnet50 --epochs 10 --batch-size 32

# 2. Train EfficientNet-B0
python -m src.train --model efficientnet_b0 --epochs 10 --batch-size 32

# 3. Train Vision Transformer (ViT-B/16)
python -m src.train --model vit_b_16 --epochs 10 --batch-size 16

# Quick dry run (1 fast epoch on small subset for verification):
python -m src.train --model resnet50 --dry-run
```

All runs, seeds, hyperparameters, and epoch metrics are automatically recorded to `results/metrics/run_log.json`.

### Run Evaluation Suite & Generate Graphs
Evaluate all models on clean and degraded test sets:
```bash
python -m src.evaluate
```
This produces:
- `results/metrics/comparison_table.csv`
- `results/metrics/clean_evaluation.json`
- `results/metrics/robustness_evaluation.json`
- `results/graphs/accuracy_f1_comparison.png`
- `results/graphs/latency_vs_f1_scatter.png`
- `results/graphs/reliability_calibration_diagram.png`

---

## 🖥️ Interactive Web Application

Launch the Streamlit app:
```bash
streamlit run app/app.py
```

### Key Application Features:
- **Single-page flow**: Upload leaf photo or capture via camera -> select plant species -> receive diagnosis, confidence score, XAI overlay, and treatment guidance.
- **Explainable AI (XAI)**: Displays side-by-side Grad-CAM / Attention Rollout heatmaps explaining the model's visual reasoning.
- **Safety Thresholding**: Flags uncertain predictions (< 60% confidence) with an advisory warning directing users to local extension experts.
- **Bilingual Localization**: Instant English ⇄ Hindi (हिन्दी) UI toggle.
- **Diagnosis History**: Automatically logs timestamped predictions and image hashes to SQLite.

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t plantcare-ai .
docker run -p 8501:8501 plantcare-ai
```
Access at `http://localhost:8501`.

---

## 🛡️ Research Disclaimer & Responsible AI

PlantCare AI is an academic research prototype developed for educational decision-support. Recommendation guidance is derived strictly from public agricultural extension sources (FAO, ICAR, Cornell, UC Davis). It does not substitute for certified in-person agronomic inspection.
