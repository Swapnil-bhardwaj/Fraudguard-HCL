"""Script to generate Project_Source_Code_Swapnil_Bhardwaj_2400320101153.docx
Consisting of all project source code, unit tests, and configuration files.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_source_code_docx(output_path: Path):
    doc = docx.Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_title.add_run("COMPLETE PROJECT SOURCE CODE LISTING\n")
    r_t.bold = True
    r_t.font.size = Pt(20)
    r_t.font.color.rgb = RGBColor(16, 24, 40)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Credit Card Fraud Detection Using Anomaly Detection\nPlatform: FraudGuard AI | Academic Year 2025–26")
    r_sub.font.size = Pt(12)
    r_sub.bold = True
    r_sub.font.color.rgb = RGBColor(79, 70, 229)

    doc.add_paragraph()

    # Meta Table
    table_data = [
        ("Project Title", "Credit Card Fraud Detection Using Anomaly Detection (Project 32)"),
        ("Learner Name", "Swapnil Bhardwaj"),
        ("Roll Number", "2400320101153"),
        ("College / Campus", "ABES Engineering College"),
        ("Live Web Application", "https://fraudguard-hcl.streamlit.app/"),
        ("GitHub Repository", "https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git"),
        ("Technology Stack", "Python 3.13, Scikit-learn, Imbalanced-learn, Streamlit, FastAPI, SQLite, Plotly"),
    ]
    
    t = doc.add_table(rows=len(table_data) + 1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.rows[0].cells[0].paragraphs[0].add_run("Field").bold = True
    t.rows[0].cells[1].paragraphs[0].add_run("Details").bold = True
    set_cell_background(t.rows[0].cells[0], "0F172A")
    set_cell_background(t.rows[0].cells[1], "0F172A")
    t.rows[0].cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    t.rows[0].cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for i, (k, v) in enumerate(table_data, start=1):
        row = t.rows[i]
        row.cells[0].paragraphs[0].add_run(k).bold = True
        r_v = row.cells[1].paragraphs[0].add_run(v)
        if "http" in v:
            r_v.font.color.rgb = RGBColor(79, 70, 229)
            r_v.bold = True
        set_cell_background(row.cells[0], "F8FAFC" if i % 2 == 1 else "FFFFFF")
        set_cell_background(row.cells[1], "F8FAFC" if i % 2 == 1 else "FFFFFF")
        row.cells[0].width = Inches(2.2)
        row.cells[1].width = Inches(4.8)

    doc.add_paragraph()

    # Architecture Overview
    h_intro = doc.add_heading("1. Source Code Manifest & Architecture Overview", level=1)
    doc.add_paragraph(
        "This document compiles the complete, functional source code for the FraudGuard AI project. "
        "The codebase encompasses the Streamlit frontend interface, FastAPI REST microservices, modular ML core pipelines "
        "(data validation, feature engineering, train-test fit discipline, Isolation Forest, Local Outlier Factor, Neural Autoencoder, "
        "SMOTE-augmented baselines, real-time scoring, audit logging), and 30 automated pytest validation suites."
    )

    code_modules = [
        # (category, filepath, description)
        ("Web Dashboard UI", "app.py", "Streamlit frontend interactive console featuring 7 operational views, high-contrast KPI cards, real-time single transaction scoring, 1-click test presets, Plotly visual analytics, and batch compliance CSV export."),
        ("REST API Microservice", "api/main.py", "FastAPI production microservice exposing /health, /predict, /predict-batch, /models, and automated Swagger OpenAPI documentation at /docs."),
        ("Data Validation Core", "src/data_validation.py", "Rigorous schema validation, categorical taxonomy enforcement, non-negative amount checks, monotonic sequence checks, and comprehensive data quality audit report generation."),
        ("Feature Engineering", "src/feature_engineering.py", "Temporal feature extraction (hour of day, day of week, cyclical sine/cosine transforms, night transactions) and rolling historical card token frequency velocity maps."),
        ("Data Preprocessing", "src/preprocessing.py", "Zero-data-leakage pipeline fitting RobustScaler (outlier-resistant median/IQR scaling) and OneHotEncoder exclusively on the 80% training partition."),
        ("Model Training Engines", "src/train_models.py", "Calibration of Isolation Forest, Local Outlier Factor (calibrated in novelty=True mode for out-of-sample scoring), Neural Autoencoder (hourglass bottleneck reconstruction error), and Supervised Logistic Regression with within-fold SMOTE."),
        ("Inference & Scoring Logic", "src/scoring.py", "Continuous anomaly score calibration, threshold evaluation, human review recommendation flags, contextual explainability indicators, and compliance disclaimer safeguards."),
        ("Evaluation Protocol", "src/evaluation.py", "Rare-event evaluation computing Precision, Recall, F1-Score, Average Precision (PR-AUC), ROC-AUC, False Positive Rate (FPR), Confusion Matrices, and microsecond latency benchmarks."),
        ("Audit Persistence", "src/persistence.py", "Joblib model pipeline serialization and SQLite audit logger (outputs/audit.db) storing execution timestamps and anomaly scores without persisting raw card numbers or PII."),
        ("Synthetic Data Utility", "src/generate_synthetic.py", "Deterministic synthetic transaction generator (4,500 rows) isolated separately in data/synthetic_transactions_only.csv with distinct provenance manifests."),
        ("Pipeline Orchestrator", "src/run_pipeline.py", "End-to-end training and evaluation workflow executing data loading, stratified 80/20 splitting, 5-model calibration, out-of-fold evaluation, artifact saving, and figure plotting."),
        ("System Utilities", "src/utils.py", "Deterministic global seed locking (Python, NumPy, Scikit-learn) and microsecond latency benchmarking timers."),
        ("Unit Tests: Validation", "tests/test_validation.py", "Pytest suite testing missing columns, corrupted timestamps, negative amounts, duplicate IDs, and schema validation integrity."),
        ("Unit Tests: Feature Engineering", "tests/test_feature_engineering.py", "Pytest suite testing temporal extraction, cyclical sine/cosine bounds (-1 to 1), and velocity features."),
        ("Unit Tests: Preprocessing", "tests/test_preprocessing.py", "Pytest suite verifying strict train-test fit discipline and unseen categorical level handling without leakage."),
        ("Unit Tests: ML Models", "tests/test_models.py", "Pytest suite testing Isolation Forest, LOF novelty=True mode, Autoencoder convergence, and within-fold SMOTE bounds."),
        ("Unit Tests: Scoring Logic", "tests/test_scoring.py", "Pytest suite verifying single record and batch prediction dictionaries, risk thresholds, and compliance flags."),
        ("Unit Tests: Metrics", "tests/test_metrics.py", "Pytest suite validating rare-event confusion matrix calculations and PR-AUC bounds under extreme class imbalance."),
        ("Unit Tests: Smoke & Integration", "tests/test_smoke_app.py", "Full integration smoke test testing Streamlit pipeline dependencies, artifact loading, and FastAPI endpoint responses."),
        ("Dependencies", "requirements.txt", "Pinned Python package dependencies for reproducible execution across environments."),
    ]

    # File Manifest Table
    manifest_table = doc.add_table(rows=len(code_modules) + 1, cols=3)
    manifest_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_headers = ["Category", "File Path", "Module Description"]
    for j, h in enumerate(m_headers):
        manifest_table.rows[0].cells[j].paragraphs[0].add_run(h).bold = True
        set_cell_background(manifest_table.rows[0].cells[j], "E2E8F0")

    for idx, (cat, fpath, desc) in enumerate(code_modules, start=1):
        row = manifest_table.rows[idx]
        row.cells[0].paragraphs[0].add_run(cat).bold = True
        r_f = row.cells[1].paragraphs[0].add_run(fpath)
        r_f.font.name = "Consolas"
        r_f.font.size = Pt(8.5)
        row.cells[2].paragraphs[0].add_run(desc)
        if idx % 2 == 1:
            set_cell_background(row.cells[0], "F8FAFC")
            set_cell_background(row.cells[1], "F8FAFC")
            set_cell_background(row.cells[2], "F8FAFC")
        row.cells[0].width = Inches(1.8)
        row.cells[1].width = Inches(2.2)
        row.cells[2].width = Inches(3.0)

    doc.add_page_break()

    # Section 2: Code Listings
    doc.add_heading("2. Complete Source Code Listings", level=1)

    for idx, (cat, fpath, desc) in enumerate(code_modules, start=1):
        p_path = Path(fpath)
        if not p_path.exists():
            print(f"File not found: {fpath}")
            continue

        code_text = p_path.read_text(encoding="utf-8")
        line_count = len(code_text.splitlines())

        # Module Header
        h_mod = doc.add_heading(f"2.{idx} {fpath} ({cat})", level=2)
        h_mod.paragraph_format.space_before = Pt(14)
        h_mod.paragraph_format.space_after = Pt(2)

        p_desc = doc.add_paragraph()
        r_desc_title = p_desc.add_run("Purpose: ")
        r_desc_title.bold = True
        p_desc.add_run(f"{desc} [Total Lines: {line_count:,}]")

        # Code Container Block (Table with shaded background)
        code_table = doc.add_table(rows=1, cols=1)
        code_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = code_table.rows[0].cells[0]
        set_cell_background(cell, "F8FAFC")
        cell.width = Inches(7.0)

        p_code = cell.paragraphs[0]
        p_code.paragraph_format.space_before = Pt(4)
        p_code.paragraph_format.space_after = Pt(4)
        p_code.paragraph_format.line_spacing = 1.05

        r_code = p_code.add_run(code_text)
        r_code.font.name = "Consolas"
        r_code.font.size = Pt(7.5)
        r_code.font.color.rgb = RGBColor(30, 41, 59)

        doc.add_paragraph()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    out_dir = Path("outputs/final_submission_pack")
    target_file = out_dir / "Project_Source_Code_Swapnil_Bhardwaj_2400320101153.docx"
    create_source_code_docx(target_file)
    # Also create the unnumbered version for convenience
    create_source_code_docx(out_dir / "Project_Source_Code_Swapnil_Bhardwaj.docx")
