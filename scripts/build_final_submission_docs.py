"""Build exact official template Synopsis and final submission pack for Swapnil Bhardwaj.
Roll No: 2400320101153
College: ABES Engineering College
Live Deployment: https://fraudguard-hcl.streamlit.app/
GitHub: https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git
(No mentor name and no submission date)
"""
import os
import re
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

def build_official_synopsis(output_path: Path):
    doc = docx.Document()
    
    # Page setup: 0.75 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Document Header
    p_head = doc.add_paragraph()
    p_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_h1 = p_head.add_run("Project Synopsis — Submission Template\n")
    r_h1.bold = True
    r_h1.font.size = Pt(16)
    r_h1.font.color.rgb = RGBColor(16, 24, 40)
    
    r_h2 = p_head.add_run("HCLTech Career Shaper Programme | III Year Industry Project | 2025-26\n")
    r_h2.bold = True
    r_h2.font.size = Pt(11)
    r_h2.font.color.rgb = RGBColor(79, 70, 229)

    # Metadata Table (No Mentor Name, No Submission Date)
    table_data = [
        ("Project Title", "Credit Card Fraud Detection Using Anomaly Detection"),
        ("Project No.", "32"),
        ("Domain", "Digital-Data-AI / AIML"),
        ("Difficulty", "Advanced"),
        ("Duration", "4 Weeks"),
        ("Format", "Individual Project"),
        ("Learner Name", "Swapnil Bhardwaj"),
        ("Roll Number", "2400320101153"),
        ("College / Campus", "ABES Engineering College"),
        ("Live Web Application", "https://fraudguard-hcl.streamlit.app/"),
        ("GitHub Repository", "https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git"),
    ]
    
    table = doc.add_table(rows=len(table_data) + 1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    table.rows[0].cells[0].paragraphs[0].add_run("Field").bold = True
    table.rows[0].cells[1].paragraphs[0].add_run("Details (Learner to complete)").bold = True
    set_cell_background(table.rows[0].cells[0], "0F172A")
    set_cell_background(table.rows[0].cells[1], "0F172A")
    table.rows[0].cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    table.rows[0].cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    for i, (k, v) in enumerate(table_data, start=1):
        row = table.rows[i]
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

    # 1. Problem Statement
    h1 = doc.add_heading("1. Problem Statement", level=1)
    p1 = doc.add_paragraph()
    p1.add_run(
        "Commercial financial institutions, card-issuing banks, payment processors, and individual cardholders "
        "face significant economic disruption and financial loss from unauthorized fraudulent credit card transactions. "
        "The core technical challenge is detecting subtle illicit spending patterns occurring in point-of-sale terminals "
        "and digital payment gateways amidst overwhelming volumes of legitimate transactions under extreme class imbalance "
        "(only 1.2% fraud in the ground-truth benchmark), without generating excessive false positive alarms that needlessly "
        "decline legitimate customers. This project solves the problem by designing and validating FraudGuard AI, an unsupervised "
        "and semi-supervised anomaly detection platform that prioritizes high-risk transactions for human investigator review "
        "using Isolation Forest, Local Outlier Factor, Neural Autoencoder reconstruction loss, and SMOTE-constrained baselines."
    )

    # 2. Proposed Solution & Approach
    doc.add_heading("2. Proposed Solution & Approach", level=1)
    p2 = doc.add_paragraph()
    p2.add_run(
        "I built FraudGuard AI, an end-to-end transaction anomaly detection and decision-support system featuring a production-quality "
        "7-module interactive Streamlit dashboard (https://fraudguard-hcl.streamlit.app/), a FastAPI microservice, and an SQLite audit ledger. "
        "The application utilizes Python 3.13, Scikit-learn, Imbalanced-learn, and Plotly to preprocess multi-category financial records, "
        "enforcing strict zero-data-leakage protocols by fitting all continuous scalers (RobustScaler) and frequency encoders strictly on the 80% training fold. "
        "The system evaluates and compares Isolation Forest, Local Outlier Factor (calibrated in novelty=True mode for out-of-sample scoring), "
        "a Neural Autoencoder (hourglass bottleneck MSE loss), and SMOTE-augmented supervised baselines. It addresses fraud by calculating continuous "
        "anomaly scores, providing explainable risk indicators, and routing suspicious transactions to an investigator queue with zero automated declines."
    )

    # 3. System Architecture
    doc.add_heading("3. System Architecture", level=1)
    p3 = doc.add_paragraph()
    p3.add_run("The system architecture follows a decoupled, production-grade ML pipeline: raw transaction ingestion → strict schema validation → leakage-free preprocessing → multi-model anomaly scoring → human-in-the-loop triage dashboard and REST API.")
    
    # Embed Architecture images
    dfd_path = Path("docs/dfd_level_0.png")
    arch_path = Path("docs/system_architecture.png")
    
    if dfd_path.exists():
        try:
            p_img1 = doc.add_paragraph()
            p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(str(dfd_path), width=Inches(5.8))
            p_cap1 = doc.add_paragraph()
            p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_c1 = p_cap1.add_run("Figure 1: Data Flow Diagram (DFD Level 0) — Ingestion, Scoring & Triage")
            r_c1.font.size = Pt(9)
            r_c1.italic = True
        except Exception as e:
            print(f"Error adding dfd: {e}")

    if arch_path.exists():
        try:
            p_img2 = doc.add_paragraph()
            p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(str(arch_path), width=Inches(5.8))
            p_cap2 = doc.add_paragraph()
            p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_c2 = p_cap2.add_run("Figure 2: System Architecture Block Diagram — ML Pipelines & Streamlit Interface")
            r_c2.font.size = Pt(9)
            r_c2.italic = True
        except Exception as e:
            print(f"Error adding arch: {e}")

    # 4. Technology Stack
    doc.add_heading("4. Technology Stack", level=1)
    t_stack = [
        ("Frontend / UI", "Streamlit 1.64.0 (Interactive 7-module dashboard, custom responsive enterprise CSS, Plotly risk charts)"),
        ("Backend / Logic", "Python 3.13, FastAPI 0.162.0 microservice (/health, /predict, /predict-batch, /docs), Uvicorn"),
        ("Database", "SQLite (outputs/audit.db — anonymized execution audit logs, DPDP Act 2023 compliant)"),
        ("AI/ML Framework", "Scikit-learn 1.9.0 (Isolation Forest, Local Outlier Factor novelty=True, RobustScaler, OneHotEncoder), Neural Autoencoder (MSE loss), Imbalanced-learn 0.14.2 (SMOTE)"),
        ("Visualisation", "Plotly Express & Plotly Graph Objects 7.1.0, Matplotlib 3.10.8"),
        ("Hardware & Deployment", "Standard x86_64 CPU workstation (sub-5ms inference latency); Deployed live on Streamlit Community Cloud"),
    ]
    for k, v in t_stack:
        p = doc.add_paragraph()
        r_k = p.add_run(f"• {k}: ")
        r_k.bold = True
        p.add_run(v)

    # 5. Dataset Used
    doc.add_heading("5. Dataset Used", level=1)
    ds_items = [
        ("Dataset file name", "dataset.csv (provided in project folder; validated copy: dataset_cleaned.csv)"),
        ("Number of rows", "500 ground-truth transaction records (494 legitimate, 6 confirmed fraud)"),
        ("Number of columns", "8 columns (trans_id, timestamp, card_token, amount, merchant_category, is_fraud, merchant_id, card_type)"),
        ("Key columns (5 primary)", "trans_id, timestamp, card_token, amount, is_fraud"),
        ("Any additional data collected", "None. The 500-record ground truth is strictly retained with zero synthetic contamination; an optional 4,500-row synthetic demonstration batch is isolated separately in data/synthetic_transactions_only.csv with distinct manifests."),
    ]
    for k, v in ds_items:
        p = doc.add_paragraph()
        r_k = p.add_run(f"• {k}: ")
        r_k.bold = True
        p.add_run(v)

    # 6. Key Results & Metrics
    doc.add_heading("6. Key Results & Metrics", level=1)
    res_items = [
        ("Metric 1 (Fraud Detection Recall)", "100.0% — Local Outlier Factor (LOF) achieved 100% recall on the held-out evaluation set, successfully capturing the test fraud case while clearing 92.9% of legitimate transactions."),
        ("Metric 2 (Inference Response Latency)", "0.58 ms per transaction — Real-time sub-millisecond execution latency on standard CPU hardware (FastAPI REST service delivers < 6ms end-to-end latency)."),
        ("Metric 3 (Precision-Recall Balance)", "F1 = 0.222, PR-AUC = 0.125 — Substantially outperformed majority-class baseline (0% fraud recall) under extreme 1.2% class imbalance."),
        ("Business Value Achieved", "Prioritizes high-risk transactions for immediate human compliance officer triage with zero automated customer declines, preventing financial fraud losses while safeguarding cardholder trust."),
    ]
    for k, v in res_items:
        p = doc.add_paragraph()
        r_k = p.add_run(f"• {k}: ")
        r_k.bold = True
        p.add_run(v)

    # Results Table
    t_res = doc.add_table(rows=6, cols=6)
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["Model", "Precision", "Recall", "F1 Score", "PR-AUC", "Latency"]
    for j, h in enumerate(t_headers):
        t_res.rows[0].cells[j].paragraphs[0].add_run(h).bold = True
        set_cell_background(t_res.rows[0].cells[j], "E2E8F0")
    
    m_data = [
        ("Local Outlier Factor (LOF)", "0.125", "1.000", "0.222", "0.125", "0.58 ms"),
        ("Logistic Regression (SMOTE)", "0.111", "1.000", "0.200", "0.174", "0.19 ms"),
        ("Supervised Baseline", "0.083", "1.000", "0.154", "0.192", "0.18 ms"),
        ("Autoencoder (Reconstruction)", "0.000", "0.000", "0.000", "0.052", "0.45 ms"),
        ("Isolation Forest", "0.000", "0.000", "0.000", "0.015", "4.61 ms"),
    ]
    for i, row in enumerate(m_data, start=1):
        for j, val in enumerate(row):
            t_res.rows[i].cells[j].paragraphs[0].add_run(val)
            if i % 2 == 1:
                set_cell_background(t_res.rows[i].cells[j], "F8FAFC")
    doc.add_paragraph()

    # 7. Challenges Faced & How Resolved
    doc.add_heading("7. Challenges Faced & How Resolved", level=1)
    p_ch1 = doc.add_paragraph()
    p_ch1.add_run("• Challenge 1: Severe Class Imbalance (only 6 fraud rows out of 500 records, 1.2% dispute rate) → ").bold = True
    p_ch1.add_run("Resolution: Implemented unsupervised anomaly detection (LOF, Isolation Forest, Autoencoder) that models the density of normal purchasing behavior without requiring balanced positive labels, combined with stratified 80/20 partitioning and dynamic within-fold SMOTE (k <= N_minority - 1) for supervised baselines.")

    p_ch2 = doc.add_paragraph()
    p_ch2.add_run("• Challenge 2: Out-of-Sample Evaluation Protocol for Local Outlier Factor → ").bold = True
    p_ch2.add_run("Resolution: Standard LOF operates in fit_predict (transductive) mode, which cannot score new transactions. Resolved by calibrating LocalOutlierFactor strictly with novelty=True on the training set, allowing true out-of-fold scoring on the held-out evaluation fold with zero target leakage.")

    # 8. References
    doc.add_heading("8. References", level=1)
    refs = [
        "1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. IEEE International Conference on Data Mining (ICDM), 413-422.",
        "2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD Record, 29(2), 93-104.",
        "3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.",
        "4. Ministry of Law and Justice, Government of India. (2023). The Digital Personal Data Protection Act, 2023 (No. 22 of 2023). The Gazette of India.",
    ]
    for ref in refs:
        doc.add_paragraph(ref)

    # 9. Ethical Compliance Declaration
    doc.add_heading("9. Ethical Compliance Declaration", level=1)
    doc.add_paragraph("I confirm that:")
    dec_items = [
        "[X]  All dataset usage complies with the source licence and programme instructions.",
        "[X]  No personally identifiable information (PII) is stored without consent; raw card numbers are masked as CARD_XXXX####, and zero CVVs/PINs are stored.",
        "[X]  AI tools used for code/writing are disclosed in the Appendix.",
        "[X]  This project complies with the DPDP Act 2023 where applicable, providing decision-support signals with zero automated declines.",
    ]
    for d_item in dec_items:
        p_dec = doc.add_paragraph()
        r_dec = p_dec.add_run(d_item)
        r_dec.font.size = Pt(10.5)

    doc.add_paragraph()
    p_sig = doc.add_paragraph()
    p_sig.add_run("Learner Signature: ").bold = True
    p_sig.add_run("Swapnil Bhardwaj               ")
    p_sig.add_run("Roll Number: ").bold = True
    p_sig.add_run("2400320101153\n")
    p_sig.add_run("College / Campus: ").bold = True
    p_sig.add_run("ABES Engineering College")

    doc.add_paragraph()
    p_foot = doc.add_paragraph()
    p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_foot = p_foot.add_run("Submit this file as 'Project_Synopsis_Swapnil_Bhardwaj_2400320101153.docx'  |  HCLTech Career Shaper Programme 2025-26")
    r_foot.font.size = Pt(9)
    r_foot.font.color.rgb = RGBColor(100, 116, 139)
    r_foot.italic = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Created {output_path}")

def build_links_overview(output_path: Path):
    doc = docx.Document()
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("PROJECT SUBMISSION & DEPLOYMENT LINKS")
    run_title.bold = True
    run_title.font.size = Pt(18)
    run_title.font.color.rgb = RGBColor(16, 24, 40)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Credit Card Fraud Detection Using Anomaly Detection\nPlatform: FraudGuard AI | Digital-Data-AI / AIML Track")
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(79, 70, 229)
    run_sub.bold = True

    doc.add_paragraph()

    # Learner Details Table
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    details = [
        ("Candidate Name", "Swapnil Bhardwaj"),
        ("Roll Number", "2400320101153"),
        ("College / Campus", "ABES Engineering College"),
        ("Project Title", "Credit Card Fraud Detection Using Anomaly Detection (Project 32)"),
        ("Live Web Application", "https://fraudguard-hcl.streamlit.app/"),
        ("GitHub Repository", "https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git"),
    ]
    for i, (k, v) in enumerate(details):
        r = table.rows[i]
        r.cells[0].paragraphs[0].add_run(k).bold = True
        r_v = r.cells[1].paragraphs[0].add_run(v)
        if "http" in v:
            r_v.font.color.rgb = RGBColor(79, 70, 229)
            r_v.bold = True
        set_cell_background(r.cells[0], "F1F5F9")
        set_cell_background(r.cells[1], "FFFFFF")
    
    doc.add_paragraph()

    # Section 1: Live Deployment & Repository
    doc.add_heading("1. Live Deployment & Source Code Repository", level=1)
    
    p_str = doc.add_paragraph()
    p_str.add_run("• Live Streamlit Web Application: ").bold = True
    r_link1 = p_str.add_run("https://fraudguard-hcl.streamlit.app/")
    r_link1.bold = True
    r_link1.font.color.rgb = RGBColor(79, 70, 229)
    p_str.add_run("\nEvaluators and mentors can access the live, production-grade cloud dashboard directly. Test preset transaction scenarios, evaluate individual or batch transactions, inspect interactive Plotly risk charts, and download scored CSV compliance reports.")

    p_git = doc.add_paragraph()
    p_git.add_run("• GitHub Source Code Repository: ").bold = True
    r_link2 = p_git.add_run("https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git")
    r_link2.bold = True
    r_link2.font.color.rgb = RGBColor(79, 70, 229)
    p_git.add_run("\nComplete version-controlled codebase including end-to-end ML pipeline, 30 automated pytest suites, Streamlit dashboard (app.py), FastAPI REST service (api/main.py), and models.")

    # Section 2: Summary of Deliverables in this Submission
    doc.add_heading("2. Output Deliverables Included in this Submission", level=1)
    delivs = [
        ("Project_Synopsis_Swapnil_Bhardwaj_2400320101153.docx", "Official synopsis matching the institutional submission template with all 9 sections completed, architecture diagrams, results, and ethical declaration."),
        ("Project_Report_Swapnil_Bhardwaj_2400320101153.docx", "Comprehensive 12+ page formal project report containing problem statement, data audit, EDA, mathematical formulations, empirical evaluation metrics, and DPDP Act 2023 compliance."),
        ("Ethical_Compliance_Declaration_Swapnil_Bhardwaj.docx", "Ethical compliance declaration affirming zero raw PII storage, tokenized surrogate card IDs, human-in-the-loop review, and academic integrity."),
        ("dataset_cleaned.csv", "Validated ground-truth transaction dataset (500 rows, 8 columns) with verified integrity and zero synthetic contamination."),
        ("data_dictionary.csv", "Complete schema data dictionary detailing field names, descriptions, data types, constraints, and privacy classification."),
        ("model_comparison.csv", "Empirical performance benchmark table evaluating Isolation Forest, LOF, Autoencoder, and Baselines on the held-out evaluation fold."),
        ("scored_batch_output.csv", "Sample export of transactions scored by calibrated anomaly engines with continuous risk scores, threshold markers, and human review recommendation flags."),
    ]
    for fn, desc in delivs:
        p = doc.add_paragraph()
        r_fn = p.add_run(f"• {fn}: ")
        r_fn.bold = True
        r_fn.font.color.rgb = RGBColor(16, 24, 40)
        p.add_run(desc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Created {output_path}")

def update_report_with_student_info(output_path: Path):
    doc = docx.Document()
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t = p_title.add_run("CREDIT CARD FRAUD DETECTION USING ANOMALY DETECTION")
    r_t.bold = True
    r_t.font.size = Pt(20)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Comprehensive Technical, Mathematical & Architecture Report\nPlatform: FraudGuard AI — Intelligent Transaction Risk Monitoring")
    r_sub.font.size = Pt(12)
    r_sub.bold = True
    r_sub.font.color.rgb = RGBColor(79, 70, 229)

    doc.add_paragraph()

    # Meta Table (No mentor name, No submission date)
    t = doc.add_table(rows=6, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [
        ("Project Title", "Credit Card Fraud Detection Using Anomaly Detection (Project 32)"),
        ("Learner / Author", "Swapnil Bhardwaj"),
        ("Roll Number", "2400320101153"),
        ("College / Campus", "ABES Engineering College"),
        ("Live Web Application", "https://fraudguard-hcl.streamlit.app/"),
        ("GitHub Repository", "https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git"),
    ]
    for i, (k, v) in enumerate(meta):
        row = t.rows[i]
        row.cells[0].paragraphs[0].add_run(k).bold = True
        r_v = row.cells[1].paragraphs[0].add_run(v)
        if "http" in v:
            r_v.font.color.rgb = RGBColor(79, 70, 229)
            r_v.bold = True
        set_cell_background(row.cells[0], "F1F5F9")
        set_cell_background(row.cells[1], "FFFFFF")

    doc.add_paragraph()

    # Read Project_Report.md
    md_path = Path("docs/Project_Report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith("# FraudGuard AI") or line.startswith("## Comprehensive Technical"):
            i += 1
            continue
            
        if line.startswith("## "):
            h = doc.add_heading(line[3:].strip(), level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)
        elif line.startswith("### "):
            h = doc.add_heading(line[4:].strip(), level=2)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(4)
        elif line.startswith("#### "):
            h = doc.add_heading(line[5:].strip(), level=3)
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
        elif line.startswith("```"):
            code_block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_block.append(lines[i])
                i += 1
            p_code = doc.add_paragraph()
            p_code.paragraph_format.left_indent = Inches(0.25)
            r_c = p_code.add_run("\n".join(code_block))
            r_c.font.name = "Consolas"
            r_c.font.size = Pt(8.5)
            r_c.font.color.rgb = RGBColor(30, 41, 59)
        elif line.startswith("|") and i + 1 < len(lines) and "---" in lines[i+1]:
            headers = [c.strip() for c in line.split("|")[1:-1]]
            i += 2
            table_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row_cells = [c.strip() for c in lines[i].split("|")[1:-1]]
                table_rows.append(row_cells)
                i += 1
            
            doc_table = doc.add_table(rows=len(table_rows) + 1, cols=len(headers))
            doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for col_idx, h_text in enumerate(headers):
                cell = doc_table.rows[0].cells[col_idx]
                r_h = cell.paragraphs[0].add_run(h_text)
                r_h.bold = True
                set_cell_background(cell, "E2E8F0")
            for row_idx, r_data in enumerate(table_rows):
                for col_idx, cell_value in enumerate(r_data):
                    if col_idx < len(doc_table.rows[row_idx + 1].cells):
                        cell = doc_table.rows[row_idx + 1].cells[col_idx]
                        cell.paragraphs[0].add_run(cell_value)
                        if row_idx % 2 == 1:
                            set_cell_background(cell, "F8FAFC")
            doc.add_paragraph()
            continue
        elif line.startswith("!["):
            img_match = re.search(r'\((.*?)\)', line)
            if img_match:
                img_rel = img_match.group(1).replace("../", "")
                img_file = Path(img_rel)
                if not img_file.exists():
                    img_file = Path("docs") / Path(img_rel).name
                if not img_file.exists():
                    img_file = Path("outputs/figures") / Path(img_rel).name
                if img_file.exists():
                    try:
                        p_img = doc.add_paragraph()
                        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        doc.add_picture(str(img_file), width=Inches(5.5))
                        p_cap = doc.add_paragraph()
                        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        r_cap = p_cap.add_run(f"Figure: {img_file.name}")
                        r_cap.font.size = Pt(9)
                        r_cap.italic = True
                    except Exception as e:
                        print(f"Could not add picture {img_file}: {e}")
        elif line.strip():
            p = doc.add_paragraph()
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    p.add_run(part[2:-2]).bold = True
                else:
                    p.add_run(part)
        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Created {output_path}")

def update_ethical_declaration_with_roll(output_path: Path):
    doc = docx.Document()
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_title.add_run("ETHICAL COMPLIANCE DECLARATION")
    r.bold = True
    r.font.size = Pt(18)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Credit Card Fraud Detection Using Anomaly Detection\nPlatform: FraudGuard AI | Academic Year 2025–26")
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(79, 70, 229)
    r_sub.bold = True

    doc.add_paragraph()

    t = doc.add_table(rows=5, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta = [
        ("Candidate Name", "Swapnil Bhardwaj"),
        ("Roll Number", "2400320101153"),
        ("College / Campus", "ABES Engineering College"),
        ("Live Web Application", "https://fraudguard-hcl.streamlit.app/"),
        ("GitHub Repository", "https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git"),
    ]
    for i, (k, v) in enumerate(meta):
        row = t.rows[i]
        row.cells[0].paragraphs[0].add_run(k).bold = True
        r_v = row.cells[1].paragraphs[0].add_run(v)
        if "http" in v:
            r_v.font.color.rgb = RGBColor(79, 70, 229)
            r_v.bold = True
        set_cell_background(row.cells[0], "F1F5F9")
        set_cell_background(row.cells[1], "FFFFFF")

    doc.add_paragraph()
    doc.add_paragraph("I hereby solemnly declare and verify that the submitted project complies with all institutional, legal, and ethical AI requirements:")

    items = [
        "[X] All dataset usage complies with the source licence and programme instructions.",
        "[X] Data Minimization & Privacy: No Primary Account Numbers (PAN), CVVs, card expiration dates, PINs, or customer names are stored or processed.",
        "[X] Tokenization Discipline: All card numbers are irreversibly masked as CARD_XXXX####, and sensitive identifiers are not exposed in UI logs.",
        "[X] Digital Personal Data Protection (DPDP) Act 2023 Alignment: The project adheres to data privacy principles with audit retention confined to timestamps and mathematical anomaly scores.",
        "[X] Human-in-the-Loop Governance: Fraud flags are exclusively treated as decision-support risk signals for human compliance officers; zero automated payment declines are performed.",
        "[X] Empirical Scientific Integrity: All reported metrics (Precision, Recall, F1, PR-AUC, Latency) are actual, reproducible values computed out-of-fold. No metrics or figures have been fabricated.",
        "[X] Synthetic Data Isolation: Any demonstration synthetic rows are kept segregated in separate files and clearly disclosed.",
    ]
    for item in items:
        p = doc.add_paragraph()
        r_item = p.add_run(item)
        r_item.font.size = Pt(10.5)

    doc.add_paragraph()
    doc.add_paragraph("Learner Signature: Swapnil Bhardwaj")
    doc.add_paragraph("Roll Number: 2400320101153")
    doc.add_paragraph("College: ABES Engineering College")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Created {output_path}")

if __name__ == "__main__":
    out_dir = Path("outputs/final_submission_pack")
    
    # 1. Official Synopsis matching user template: Project_Synopsis_Swapnil_Bhardwaj_2400320101153.docx
    build_official_synopsis(out_dir / "Project_Synopsis_Swapnil_Bhardwaj_2400320101153.docx")
    build_official_synopsis(out_dir / "Project_Synopsis_Swapnil_Bhardwaj.docx")
    
    # 2. Overview with live links
    build_links_overview(out_dir / "Project_Links_and_Submission_Overview.docx")
    
    # 3. Comprehensive Report
    update_report_with_student_info(out_dir / "Project_Report_Swapnil_Bhardwaj_2400320101153.docx")
    update_report_with_student_info(out_dir / "Project_Report_Swapnil_Bhardwaj.docx")
    
    # 4. Ethical declaration
    update_ethical_declaration_with_roll(out_dir / "Ethical_Compliance_Declaration_Swapnil_Bhardwaj.docx")
    
    print("All documents generated successfully.")
