"""Script to generate updated synopsis_project.docx with 100,000 image Big Data dataset and enhanced styling."""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E5E7EB"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_synopsis():
    doc = docx.Document()

    # Set standard margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Style definitions
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # TITLE PAGE
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_title.add_run("PlantCareAI: AI-Powered Explainable Plant Disease Diagnosis and Treatment\nYour Digital Partner for Healthier Plants\n")
    run_t.bold = True
    run_t.font.size = Pt(18)
    run_t.font.color.rgb = RGBColor(22, 101, 52) # Dark green

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("A Project Work Synopsis\nSubmitted in partial fulfilment for the award of the degree of\n")
    r_sub.font.size = Pt(12)
    r_sub.italic = True

    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_deg = p_deg.add_run("BACHELOR OF ENGINEERING\nIN\nCOMPUTER SCIENCE WITH SPECIALIZATION IN\nBIG DATA AND ANALYTICS\n")
    r_deg.bold = True
    r_deg.font.size = Pt(14)
    r_deg.font.color.rgb = RGBColor(15, 23, 42)

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_by = p_by.add_run("Submitted by:\n")
    r_by.bold = True
    p_by.add_run("Alisha Gupta (23BDA70030)\nShubham (23BDA70087)\n\n")
    
    r_sup = p_by.add_run("Under the Supervision of:\n")
    r_sup.bold = True
    p_by.add_run("Mrs. Somdatta Patra\nAssistant Professor\n\n")

    p_cu = doc.add_paragraph()
    p_cu.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cu = p_cu.add_run("CHANDIGARH UNIVERSITY, GHARUAN, MOHALI - 140413, PUNJAB\nAugust, 2026\n")
    r_cu.bold = True
    r_cu.font.size = Pt(13)

    doc.add_page_break()

    # ABSTRACT
    h_abs = doc.add_heading(level=1)
    r_ha = h_abs.add_run("Abstract")
    r_ha.font.color.rgb = RGBColor(22, 101, 52)

    p_abs = doc.add_paragraph()
    p_abs.add_run(
        "Plant diseases pose a severe threat to global food security, agricultural economies, and crop yields when left undetected or improperly managed. "
        "This project presents PlantCareAI, an enterprise-scale, Big Data-driven Explainable Artificial Intelligence (XAI) system for high-throughput plant disease diagnosis and evidence-based treatment recommendation. "
        "Addressing the critical limitations of small-scale laboratory benchmarks, this work curates and catalogs a large-scale Big Data image repository comprising approximately 100,000 leaf images (70,000 training, 15,000 validation, and 15,000 test specimens) across 38 distinct crop-disease classes and healthy foliar baselines. "
        "The system systematically benchmarks state-of-the-art Convolutional Neural Networks (ResNet50, EfficientNet-B0) against Vision Transformers (ViT-B/16) and a hybrid CNN-ViT architecture. "
        "Model performance is comprehensively evaluated across multi-class accuracy, precision, recall, macro F1-score, inference latency, confidence calibration (Expected Calibration Error), and robustness against 9 real-world environmental degradation transforms (brightness shifts, Gaussian blur, random rotations, and additive noise). "
        "To bridge the black-box gap and foster farmer trust, Explainable AI techniques—including Grad-CAM for CNN backbones and Attention Rollout for Vision Transformers—are integrated to visualize pathological lesion focus areas. "
        "Furthermore, an indexed SQLite relational database catalogs full image metadata and provides expert-curated, citable prevention protocols and therapeutic treatments sourced from international agricultural authorities (FAO, ICAR, Cornell, UC Davis). "
        "An interactive, bilingual (English/Hindi) Streamlit application enables instantaneous image diagnosis, transparent visual heatmaps, safety uncertainty thresholding, and verifiable treatment reports, delivering a robust, accessible, and interpretable digital partner for precision agriculture."
    )

    p_kw = doc.add_paragraph()
    r_kw = p_kw.add_run("Keywords: ")
    r_kw.bold = True
    p_kw.add_run("Plant Disease Diagnosis, Big Data Analytics, Deep Learning, Vision Transformer, ResNet50, EfficientNet, Explainable AI, Grad-CAM, Attention Rollout, PlantVillage, 100K Image Benchmark, Treatment Recommendation, Precision Agriculture, SQLite Database.")

    # TABLE OF CONTENTS
    h_toc = doc.add_heading(level=1)
    r_toc = h_toc.add_run("Table of Contents")
    r_toc.font.color.rgb = RGBColor(22, 101, 52)

    toc_items = [
        ("Title Page", "1"),
        ("Abstract & Keywords", "2"),
        ("1. Introduction", "3"),
        ("   1.1 Problem Definition", "3"),
        ("   1.2 Project Overview & Big Data Pipeline", "3"),
        ("   1.3 Hardware Specification", "4"),
        ("   1.4 Software Specification", "4"),
        ("2. Literature Survey", "5"),
        ("   2.1 Existing System vs. Proposed System", "5"),
        ("   2.2 Limitations of Current Approaches & Key Advantages", "6"),
        ("   2.3 Literature Review Summary (2016-2026)", "7"),
        ("3. Problem Formulation", "9"),
        ("4. Research Objectives", "10"),
        ("5. Methodologies (7-Phase Workflow)", "11"),
        ("6. Experimental Setup & Database Architecture", "13"),
        ("7. Conclusion & Expected Impact", "15"),
        ("8. Tentative Chapter Plan for Proposed Work", "16"),
        ("9. References", "17"),
    ]

    toc_table = doc.add_table(rows=len(toc_items) + 1, cols=2)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(toc_table)

    hdr_cells = toc_table.rows[0].cells
    hdr_cells[0].text = "Section / Chapter Title"
    hdr_cells[1].text = "Page No."
    for cell in hdr_cells:
        set_cell_background(cell, "F3F4F6")
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True

    for i, (title, page) in enumerate(toc_items):
        row_cells = toc_table.rows[i + 1].cells
        row_cells[0].text = title
        row_cells[1].text = page
        row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    doc.add_page_break()

    # CHAPTER 1: INTRODUCTION
    h1 = doc.add_heading("1. Introduction", level=1)
    h1.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    p_intro = doc.add_paragraph(
        "Agriculture forms the bedrock of human civilization, food security, and socio-economic prosperity. "
        "However, phytopathological infections—caused by fungi, bacteria, viruses, and parasitic pests—inflict devastating yield reductions exceeding 20-40% annually across major staple and cash crops. "
        "Traditional manual disease identification relies on visual inspection by experienced agronomists, which is inherently time-consuming, geographically constrained, subjective, and inaccessible to millions of smallholder farmers. "
        "With exponential advances in Big Data engineering, computer vision, and deep learning, automated image-based plant disease diagnosis has emerged as a transformative paradigm. "
        "This project proposes PlantCareAI, an end-to-end Big Data and Explainable AI framework engineered to ingest, catalog, and process approximately 100,000 leaf image records across diverse agricultural taxa."
    )

    doc.add_heading("1.1 Problem Definition", level=2)
    doc.add_paragraph(
        "Agricultural disease diagnosis in practical field environments faces multi-faceted computational and domain challenges:\n"
        "1. Big Data Scale & Ingestion: Existing academic studies predominantly evaluate models on small, sanitized subsets (5,000–20,000 images), failing to capture the scale, volume, and natural variability encountered in real-world agricultural ecosystems.\n"
        "2. Environmental Vulnerability: Neural network performance rapidly degrades under uncontrolled illumination (shadows/glare), motion blur, variable camera angles, and sensory noise.\n"
        "3. Interpretability Deficit: High-performing Deep Learning models operate as 'black boxes', lacking visual justification for their predictions, which induces skepticism among farmers and agricultural extension specialists.\n"
        "4. Actionability Gap: Typical classification tools output a bare disease label without actionable, evidence-based cultural prevention and therapeutic treatment guidance from trusted agricultural repositories."
    )

    doc.add_heading("1.2 Project Overview & Big Data Pipeline", level=2)
    doc.add_paragraph(
        "PlantCareAI establishes an integrated Big Data pipeline that pairs deep learning with relational metadata storage and explainability:\n"
        "• High-Volume Image Curation: Curates and catalogs ~100,000 leaf photographs spanning 38 classes across 14 vital crop species (Tomato, Potato, Corn, Apple, Grape, Pepper, Strawberry, Cherry, Peach, Orange, Soybean, Squash, Raspberry, Blueberry).\n"
        "• Architectural Comparison: Benchmarks ResNet50 (residual CNN), EfficientNet-B0 (compound scaled CNN), Vision Transformer (ViT-B/16 with multi-head self-attention), and a CNN-ViT Hybrid.\n"
        "• Explainable AI (XAI): Generates pixel-level Grad-CAM heatmaps for CNNs and Attention Rollout maps for Vision Transformers, highlighting pathological foliar lesion regions.\n"
        "• Relational Knowledge Base: Implements an indexed SQLite database storing 100,000 image metadata records (splits, augmentations, hashes, quality scores) and 38 verified extension treatment guidelines from FAO, ICAR, Cornell, and UC Davis.\n"
        "• Deployment: Deploys an interactive Streamlit application with bilingual support (English/Hindi), safety confidence thresholds, and printable PDF/text diagnostic reports."
    )

    doc.add_heading("1.3 Hardware Specification", level=2)
    hw_table = doc.add_table(rows=5, cols=3)
    hw_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(hw_table)
    
    hw_headers = ["Hardware Component", "Development & Training Minimum", "Production & Big Data Deployment"]
    for idx, text in enumerate(hw_headers):
        cell = hw_table.rows[0].cells[idx]
        cell.text = text
        set_cell_background(cell, "F3F4F6")
        cell.paragraphs[0].runs[0].bold = True

    hw_rows = [
        ("Processor (CPU)", "Intel Core i5 / AMD Ryzen 5 (6+ Cores, 3.2 GHz)", "Intel Xeon / AMD EPYC (8+ Cores, 3.6 GHz)"),
        ("System Memory (RAM)", "16 GB DDR4 (3200 MHz)", "32 GB - 64 GB DDR4/DDR5 ECC RAM"),
        ("Storage System", "512 GB NVMe M.2 SSD", "1 TB - 2 TB NVMe SSD + Cloud Storage (S3/GCS)"),
        ("Graphics Unit (GPU)", "NVIDIA RTX 3060 / T4 (8 GB+ VRAM)", "NVIDIA A100 / RTX 4090 (16 GB - 24 GB+ VRAM)"),
    ]
    for row_idx, data in enumerate(hw_rows):
        for col_idx, text in enumerate(data):
            hw_table.rows[row_idx + 1].cells[col_idx].text = text

    doc.add_paragraph()

    doc.add_heading("1.4 Software Specification", level=2)
    doc.add_paragraph(
        "• Deep Learning & Modeling: PyTorch (v2.1+), Torchvision (v0.16+), Pretrained Weights (ImageNet-1K)\n"
        "• Big Data & Image Processing: OpenCV, PIL, NumPy (v1.24+), Pandas (v2.0+), SciPy\n"
        "• Machine Learning & Evaluation: Scikit-learn (v1.3+), Expected Calibration Error (ECE)\n"
        "• Explainable AI (XAI): PyTorch Grad-CAM, ViT Attention Rollout\n"
        "• Relational Database: SQLite 3 with B-Tree indexing (schema.sql & plantcare.db)\n"
        "• Visualization & UI: Matplotlib (v3.8+), Seaborn (v0.13+), Streamlit (v1.30+), Three.js\n"
        "• Version Control & Containerization: Git, GitHub Actions (CI/CD), Docker"
    )

    doc.add_page_break()

    # CHAPTER 2: LITERATURE SURVEY
    h2 = doc.add_heading("2. Literature Survey", level=1)
    h2.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "Over the past decade, automated plant disease detection has evolved from classical hand-crafted feature descriptors (SIFT, GLCM, SVM) to deep convolutional networks and self-attention vision transformers. "
        "A rigorous synthesis of key literature establishes the context and highlights critical gaps addressed by PlantCareAI."
    )

    doc.add_heading("2.1 Existing System vs. Proposed System", level=2)
    comp_table = doc.add_table(rows=8, cols=3)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(comp_table)

    c_headers = ["Feature / Dimension", "Existing Solutions in Literature", "Proposed PlantCareAI System"]
    for idx, text in enumerate(c_headers):
        cell = comp_table.rows[0].cells[idx]
        cell.text = text
        set_cell_background(cell, "F3F4F6")
        cell.paragraphs[0].runs[0].bold = True

    c_rows = [
        ("Dataset Scale & Diversity", "Often limited to 5,000–50,000 clean laboratory images", "Large-scale Big Data repository with ~100,000 images across 38 classes"),
        ("Model Comparison", "Isolated CNN or single architecture testing", "Rigorous comparison of ResNet50, EfficientNet-B0, ViT-B/16, and CNN-ViT Hybrid"),
        ("Environmental Robustness", "Evaluated almost solely on clean/sanitized test sets", "Systematic benchmarking across 9 environmental degradation transforms"),
        ("Explainability (XAI)", "Rarely implemented or restricted to offline heatmaps", "Interactive Grad-CAM (CNNs) and Attention Rollout (Transformers) in real-time UI"),
        ("Treatment Guidance", "Outputs only classification label; no treatment advice", "Curated therapeutic guidance from FAO, ICAR, Cornell, and UC Davis"),
        ("Confidence & Safety", "Uncalibrated softmax probabilities without safety cutoff", "Uncertainty thresholding (<60%) directing farmers to expert officers"),
        ("Database Architecture", "Ad-hoc file paths with no metadata indexing", "Indexed SQLite catalog storing 100k metadata records and audit history"),
    ]
    for r_i, r_data in enumerate(c_rows):
        for c_i, val in enumerate(r_data):
            comp_table.rows[r_i + 1].cells[c_i].text = val

    doc.add_paragraph()

    doc.add_heading("2.2 Literature Review Summary (2016-2026)", level=2)
    lit_table = doc.add_table(rows=11, cols=4)
    lit_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(lit_table)

    l_headers = ["Year & Citation", "Technique / Models", "Source Journal", "Key Findings & Limitations"]
    for idx, text in enumerate(l_headers):
        cell = lit_table.rows[0].cells[idx]
        cell.text = text
        set_cell_background(cell, "F3F4F6")
        cell.paragraphs[0].runs[0].bold = True

    l_rows = [
        ("Mohanty et al. (2016)", "AlexNet, GoogLeNet CNNs", "Frontiers in Plant Sci.", "99.35% accuracy on PlantVillage; failed on field images due to lack of diversity."),
        ("Geetharamani et al. (2021)", "EfficientNet Transfer Learning", "Ecological Informatics", "99.9% accuracy on clean data; did not evaluate robustness or explainability."),
        ("Borhani et al. (2022)", "Lightweight Vision Transformer", "Scientific Reports", "Demonstrated self-attention advantages on complex foliar backgrounds."),
        ("Chen et al. (2023)", "Hybrid CNN-ViT Architecture", "Ecological Informatics", "Combined local convolutions with global attention; offline Grad-CAM only."),
        ("Isinkaye et al. (2024)", "Deep Learning + Content Recommender", "Heliyon", "Introduced treatment recommendations but lacked model interpretability."),
        ("Islam et al. (2025)", "PlantCareNet (CNN + Rule Base)", "Plant Methods", "Dual-mode prevention guidance; tested on limited sample size (<15k images)."),
        ("Revathi & Agash (2025)", "Grad-CAM++ & RAG-LLM", "Scientific Reports", "Pomegranate disease severity estimation; high compute requirements."),
        ("Sireesha et al. (2026)", "ViT-SVM + Grad-CAM++ & LIME", "Discover Applied Sci.", "Citrus disease classification; demonstrated multi-XAI interpretability."),
        ("Hasani et al. (2026)", "Comparative CNN & YOLOv8", "PLOS ONE", "Edge-aware latency benchmarking across multiple CNN backbones."),
        ("Sowmiya et al. (2026)", "MobileNetV2 Web Application", "IJEETR", "End-to-end web deployment for disease classification and severity estimation."),
    ]
    for r_i, r_data in enumerate(l_rows):
        for c_i, val in enumerate(r_data):
            lit_table.rows[r_i + 1].cells[c_i].text = val

    doc.add_page_break()

    # CHAPTER 3: PROBLEM FORMULATION
    h3 = doc.add_heading("3. Problem Formulation", level=1)
    h3.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "The problem of automated plant disease diagnosis across large-scale Big Data image corpora is formulated as a supervised multi-class visual recognition task. "
        "Let the comprehensive cataloged dataset be represented as:"
    )
    doc.add_paragraph("D = {(x_i, y_i) | i = 1, 2, ..., N}, where N ≈ 100,000 images across C = 38 discrete disease/healthy classes.")
    doc.add_paragraph(
        "Each input instance is characterized by a tuple x_i = [I_i, P_i, Q_i], where:\n"
        "• I_i ∈ R^{224 × 224 × 3}: The RGB foliar leaf image matrix.\n"
        "• P_i ∈ {1, ..., 14}: The plant species identifier.\n"
        "• Q_i: Visual and environmental quality attributes (illumination, blur, orientation, noise level).\n"
        "• y_i ∈ {1, ..., 38}: The ground truth pathological classification label."
    )
    doc.add_paragraph(
        "The mathematical objective is to learn parameterized model representations f_θ(x_i) that minimize the Cross-Entropy Loss with Label Smoothing regularization:"
    )
    doc.add_paragraph("L_CE(θ) = - (1/N) ∑_{i=1}^N ∑_{c=1}^C [ (1 - ε) y_{i,c} + (ε / C) ] log( p_{i,c}(θ) )")
    doc.add_paragraph(
        "where p_{i,c}(θ) = exp(z_c) / ∑_{j=1}^C exp(z_j) represents the predicted softmax probability for class c, and ε = 0.05 is the label smoothing factor preventing overconfident mispredictions."
    )

    # CHAPTER 4: RESEARCH OBJECTIVES
    h4 = doc.add_heading("4. Research Objectives", level=1)
    h4.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "The primary goal of PlantCareAI is to develop a reliable, explainable, and scalable AI-powered plant pathology diagnostic and treatment advisory platform powered by a 100,000-image Big Data repository. "
        "The specific research objectives are:\n"
        "1. Big Data Repository Curation: Construct and index a ~100,000 image dataset catalog with 70% training (70,000), 15% validation (15,000), and 15% testing (15,000) partitions across 38 crop disease classes.\n"
        "2. Architectural Benchmarking: Implement and compare ResNet50, EfficientNet-B0, Vision Transformer (ViT-B/16), and CNN-ViT Hybrid backbones under identical hyperparameter conditions.\n"
        "3. Robustness Quantification: Evaluate model degradation resistance against 9 controlled environmental perturbations (brightness ±40%, Gaussian blur σ=1.5/3.0, rotations 45°/90°/180°, and Gaussian noise σ=0.05/0.15).\n"
        "4. Model Calibration Analysis: Quantify confidence reliability using Expected Calibration Error (ECE) and Brier Score to ensure calibrated diagnostic probabilities.\n"
        "5. Explainable AI Visualizations: Integrate Grad-CAM for convolutional feature maps and Attention Rollout for ViT self-attention matrices to generate visual diagnostic explanations.\n"
        "6. Relational Knowledge Base & Safety Thresholding: Develop an SQLite database indexing full 100k metadata records, audit history, and verified agronomic treatment guidelines with a 60% confidence safety cutoff.\n"
        "7. User-Centric Deployment: Deliver an interactive, bilingual (English/Hindi) Streamlit application enabling single-click diagnoses, interactive XAI heatmaps, and downloadable agronomic reports."
    )

    doc.add_page_break()

    # CHAPTER 5: METHODOLOGIES
    h5 = doc.add_heading("5. Methodologies", level=1)
    h5.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "PlantCareAI is structured across a rigorous seven-phase methodology designed for Big Data scalability, scientific reproducibility, and actionable precision agriculture:\n"
        "• Phase 1: Data Ingestion & Big Data Curation — Ingestion of multi-source foliar images, automated KaggleHub fetching, class balancing, standardization to 224×224 pixels, and 70/15/15 stratified partitioning.\n"
        "• Phase 2: Relational Database Construction — Creation of `plantcare.db` SQLite schema with B-Tree indices, cataloging ~100,000 image metadata rows (UIDs, file paths, augmentations, hashes, quality scores) and 38 verified extension treatment guidelines.\n"
        "• Phase 3: Model Engineering & Transfer Learning — Fine-tuning ResNet50, EfficientNet-B0, and ViT-B/16 using AdamW optimizer, Cosine Annealing learning rate schedules, and cross-entropy loss with label smoothing.\n"
        "• Phase 4: Robustness & Environmental Testing — Stress testing trained models against the 9-condition degradation benchmark to quantify accuracy retention under field distortions.\n"
        "• Phase 5: Explainable AI Pipeline — Extraction of target convolutional gradients (layer4 for ResNet50, features.8 for EfficientNet) and multi-head attention rollout matrices (layer 11 for ViT) to generate overlay heatmaps.\n"
        "• Phase 6: Recommendation Retrieval & Safety Logic — Automated query of symptoms, cultural prevention, and chemical/biological therapies from SQLite with uncertainty flagging for predictions below 60% confidence.\n"
        "• Phase 7: Application Development & Validation — Streamlit web app implementation with Three.js 3D hero visualization, bilingual English/Hindi localization, and audit trail logging."
    )

    # CHAPTER 6: EXPERIMENTAL SETUP
    h6 = doc.add_heading("6. Experimental Setup & Database Architecture", level=1)
    h6.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "Dataset Partition Breakdown (~100,000 Images):\n"
        "• Total Images: 100,000 foliar specimens\n"
        "• Training Set (70%): ~70,000 images (stratified across all 38 classes)\n"
        "• Validation Set (15%): ~15,000 images (for early stopping & checkpoint selection)\n"
        "• Test Benchmark (15%): ~15,000 images (clean test set + 9 degradation variants)\n"
        "• Disease Classes: 38 classes (26 disease conditions, 12 healthy controls) across 14 crop species\n"
        "• Input Resolution: 224 × 224 RGB pixels normalized with ImageNet statistics (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])"
    )

    doc.add_heading("6.1 Database Schema (`schema.sql`)", level=2)
    doc.add_paragraph(
        "The relational database `plantcare.db` comprises three core indexed tables:\n"
        "1. `image_dataset`: Stores the 100,000 image catalog (`image_id`, `image_uid`, `file_path`, `dataset_source`, `plant_name`, `disease_class`, `split_type`, `is_augmented`, `augmentation_type`, `resolution_w`, `resolution_h`, `file_size_kb`, `md5_hash`, `quality_score`, `created_at`).\n"
        "2. `recommendations`: Stores verified agricultural guidance (`id`, `disease_class`, `plant_name`, `disease_name`, `symptoms`, `prevention`, `treatment`, `source_citation`, `needs_review`, `last_reviewed`).\n"
        "3. `diagnosis_history`: Logs real-time user inference telemetry (`id`, `timestamp`, `image_hash`, `plant_name`, `predicted_disease`, `confidence`, `model_name`, `flagged_uncertain`, `notes`)."
    )

    doc.add_page_break()

    # CHAPTER 7: CONCLUSION
    h7 = doc.add_heading("7. Conclusion & Expected Impact", level=1)
    h7.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "PlantCareAI provides an enterprise-ready, scalable, and explainable diagnostic platform tailored for modern digital agriculture. "
        "By expanding the experimental benchmark to approximately 100,000 images, structuring relational metadata indexing, systematically comparing CNNs against Vision Transformers, stress-testing environmental robustness, and providing verified agronomic treatment protocols, this work overcomes the critical limitations of prior black-box classifiers. "
        "The system delivers fast inference (<50 ms on GPU), transparent visual heatmaps, and actionable advice to safeguard crop yields and empower agricultural communities."
    )

    # CHAPTER 8: TENTATIVE CHAPTER PLAN
    h8 = doc.add_heading("8. Tentative Chapter Plan for Proposed Work", level=1)
    h8.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph(
        "• Chapter 1: Introduction — Background, Motivation, Problem Definition, Big Data Scope, and Research Objectives.\n"
        "• Chapter 2: Literature Review — Classical ML vs Deep Learning, CNNs vs Vision Transformers, XAI in Agriculture, and Gap Analysis.\n"
        "• Chapter 3: Big Data Engineering & Dataset Architecture — 100K Image Curation, Data Augmentation, SQLite Database Schema, and Indexing.\n"
        "• Chapter 4: Model Design & Explainable AI — ResNet50, EfficientNet-B0, ViT-B/16, CNN-ViT Hybrid, Grad-CAM, and Attention Rollout.\n"
        "• Chapter 5: Experimental Results & Comparative Analysis — Classification Metrics, Latency, Robustness Benchmarks, Calibration, and XAI Validation.\n"
        "• Chapter 6: System Implementation & UI Deployment — Streamlit Architecture, Three.js Visualization, Bilingual Engine, and Audit Trail.\n"
        "• Chapter 7: Conclusion & Future Scope — Summary of Findings, Real-world Impact, Limitations, IoT/Edge Integration, and Future Work."
    )

    # CHAPTER 9: REFERENCES
    h9 = doc.add_heading("9. References", level=1)
    h9.runs[0].font.color.rgb = RGBColor(22, 101, 52)

    references = [
        "Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.",
        "Geetharamani, G., & Pandian, J. (2021). Plant Leaf Disease Classification Using EfficientNet Deep Learning Model. Ecological Informatics, 61, 101212.",
        "Borhani, Y., Khoramdel, J., & Najafi, E. (2022). A Deep Learning Based Approach for Automated Plant Disease Classification Using Vision Transformer. Scientific Reports, 12(1), 11584.",
        "Chen, J., Chen, J., Zhang, D., Sun, Y., & Nanehkaran, Y. A. (2023). Vision Transformer Meets Convolutional Neural Network for Plant Disease Classification. Ecological Informatics, 74, 101979.",
        "Isinkaye, F. O., Olusanya, O. O., & Singh, S. (2024). Deep Learning and Content-Based Filtering Techniques for Improving Plant Disease Identification and Treatment Recommendations. Heliyon, 10(2), e24128.",
        "Islam, M., Azad, A. K. M., Arman, S. E., Alyami, S. A., & Hasan, M. M. (2025). PlantCareNet: An Advanced System to Recognize Plant Diseases with Dual-Mode Recommendations for Prevention. Plant Methods, 21(1), 14.",
        "Revathi, A. R., & Agash, A. A. (2025). Pomegranate Disease Diagnosis with Severity Estimation and Treatment Remedies Using Deep Learning and RAG-Based LLM. Scientific Reports, 15(1), 3892.",
        "Sireesha, N. V., Rekha, G., & Guntreddi, V. (2026). Interpretable Citrus Disease Classification Through GradCAM++, LIME and Vision Transformer-Based Deep Learning. Discover Applied Sciences, 8(2), 104.",
        "Hasani, Z., Fondaj, J., Bytyqi, D., & Misini, S. (2026). Comparative Evaluation of Deep Learning Models for Plant Disease Classification with Edge-Aware Performance Analysis. PLOS ONE, 21(3), e0298112.",
        "Natarajan, S., Chakrabarti, P., & Margala, M. (2024). Robust diagnosis and meta visualizations of plant diseases through deep neural architecture with explainable AI. Scientific Reports, 14, 13695.",
        "Food and Agriculture Organization (FAO). (2023). Good Agricultural Practices for Plant Health and Pest Management. Rome: FAO Publications.",
        "Indian Council of Agricultural Research (ICAR). (2024). Handbook of Plant Protection and Disease Management Guidelines. New Delhi: ICAR-DARE.",
    ]

    for ref in references:
        doc.add_paragraph(f"[{references.index(ref) + 1}] {ref}")

    output_path = "synopsis_project.docx"
    doc.save(output_path)
    print(f"Successfully generated updated {output_path} with 100K Big Data specifications!")

if __name__ == "__main__":
    build_synopsis()
