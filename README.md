# FraudGuard AI — Enterprise Credit Card Fraud Anomaly Detection Platform
### Advanced Transaction Risk Intelligence & Human-in-the-Loop Triage
**Author:** Swapnil Bhardwaj | **Repository:** `fraudguard-anomaly`

---

## 📌 Executive Summary
This repository delivers a complete, runnable, and thoroughly tested academic prototype for credit card transaction anomaly detection under severe class imbalance ($1.2\%$ fraud in ground truth). The project compares five modeling strategies:
1. **Isolation Forest** (Path length isolation ensemble)
2. **Local Outlier Factor (LOF)** (Density-based out-of-sample novelty detection)
3. **Neural Autoencoder** (Hourglass bottleneck Multi-Layer Perceptron measuring Mean Squared Error reconstruction loss)
4. **Supervised Baseline** (Class-weighted Logistic Regression)
5. **Supervised Baseline with SMOTE** (Resampling strictly inside the training fold with dynamic $k$-nearest neighbors)

The application enforces a **Human-in-the-Loop Decision Support** paradigm: flagged transactions serve as risk signals for human compliance officers rather than triggering automated payment declines or account freezes.

---

## 📂 Repository Structure

```text
fraud-detection-anomaly/
├── app.py                           # 7-tab interactive Streamlit web dashboard
├── requirements.txt                 # Complete project dependencies
├── .gitignore                       # Clean Git configuration
├── README.md                        # Master project guide
│
├── api/                             # FastAPI REST API Service
│   ├── __init__.py
│   └── main.py                      # Endpoints: /health, /models, /predict, /predict-batch
│
├── src/                             # Core modular machine-learning engineering
│   ├── __init__.py
│   ├── data_validation.py           # Schema, range, null, category & type validation
│   ├── feature_engineering.py       # Cyclical hour/day, log1p amount, prior card velocity
│   ├── preprocessing.py             # RobustScaler & OneHotEncoder (fit on train fold only)
│   ├── train_models.py              # Stratified partitioning & model training algorithms
│   ├── scoring.py                   # Inverted uniform anomaly scoring & risk indicators
│   ├── evaluation.py                # Precision, Recall, F1, PR-AUC, ROC-AUC, FPR & latency
│   ├── persistence.py               # Model serialization & SQLite audit logger
│   ├── utils.py                     # Safe logging, seed management & timing
│   ├── clean_data.py                # Dataset cleaning & normalization script
│   ├── eda.py                       # Exploratory Data Analysis script
│   └── generate_synthetic.py        # Isolated demonstration dataset generator
│
├── tests/                           # Pytest automated test suite (30 passing tests)
│   ├── __init__.py
│   ├── test_validation.py           # Schema assertions & error handling
│   ├── test_feature_engineering.py  # Leakage guards & temporal transformations
│   ├── test_preprocessing.py        # Out-of-fold generalization & unknown category handling
│   ├── test_models.py               # Model fitting, thresholding & SMOTE edge cases
│   ├── test_scoring.py              # Uniform score orientation & batch predictions
│   ├── test_metrics.py              # Imbalanced metrics & single-class boundary cases
│   └── test_smoke_app.py            # API endpoint verification
│
├── data/                            # Transaction datasets
│   ├── dataset.csv                  # Original immutable 500 records (ground truth)
│   ├── dataset_cleaned.csv          # Standardized 500 records
│   ├── data_dictionary.csv          # Completed field descriptions & types
│   ├── synthetic_transactions_only.csv # 4,500 isolated demonstration records
│   └── synthetic_data_manifest.json # Synthetic data audit manifest
│
├── models/                          # Persisted model pipelines & JSON metadata
│   └── v1/                          # Version 1 serialized artifacts (.joblib)
│
├── outputs/                         # Run outputs & diagnostic charts
│   ├── audit.db                     # SQLite anonymized audit log
│   ├── figures/                     # EDA charts, confusion matrices, comparison plots
│   ├── metrics/                     # evaluation_results.json & model_comparison.csv
│   └── scored_transactions/         # Downloadable compliance CSV exports
│
└── docs/                            # Formal academic documentation
    ├── Project_Report.md            # Comprehensive 10-page academic technical report
    ├── Project_Synopsis.md          # Formatted programme synopsis
    ├── Ethical_Compliance_Declaration.md # Signed ethical compliance declaration
    ├── demo_script.md               # 1–2 minute live demonstration script
    ├── system_architecture.png      # High-resolution architectural diagram
    ├── dfd_level_0.png              # Data Flow Diagram Level 0
    ├── Project_Report_Draft_Swapnil_Bhardwaj.docx # Populated report document
    └── Project_Synopsis_Swapnil_Bhardwaj_ROLLNO_PENDING.docx # Populated synopsis document
```

---

## ⚙️ Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13.7)
- Git and standard terminal (PowerShell, Bash, or Zsh)

### Step 1: Clone Repository / Navigate to Workspace
```bash
cd C:\Users\Spell\.gemini\antigravity\scratch\fraud-detection-anomaly
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1. Run Automated Test Suite (Quality Gate)
Execute the complete pytest suite:
```bash
python -m pytest tests/ -v
```
*Expected Result:* **30 passed, 0 failed** in ~7 seconds.

### 2. Run Data Cleaning & Exploratory Data Analysis
```bash
# Normalize provided dataset (immutable ground truth preserved)
python src/clean_data.py --input data/dataset.csv --output data/dataset_cleaned.csv

# Generate EDA charts and JSON summary
python src/eda.py --input data/dataset_cleaned.csv --output-dir outputs
```

### 3. Train & Evaluate All Models
Execute the end-to-end ML lifecycle:
```bash
python -m src.run_pipeline --dataset data/dataset_cleaned.csv
```
This trains Isolation Forest, LOF, Autoencoder, and Supervised Baselines with SMOTE, evaluates them on the held-out test split, exports metrics to `outputs/metrics/`, generates confusion matrices, and persists models to `models/v1/`.

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Launch FastAPI REST Service (Optional)
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 📊 Measured Model Performance (Held-Out Test Set)

Evaluated on an $80/20$ stratified holdout (100 unseen transactions: 99 legitimate, 1 fraud):

| Model | Decision Threshold | TP | FP | TN | FN | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | FPR | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Local Outlier Factor (LOF)** | $1.14289$ | **1** | 7 | 92 | **0** | **12.50%** | **100.0%** | **0.2222** | 0.2000 | 0.9596 | 7.07% | 5.62 ms |
| **Isolation Forest** | $0.57542$ | 0 | **1** | **98** | 1 | 0.00% | 0.0% | 0.0000 | **0.5000** | **0.9899** | **1.01%** | 11.95 ms |
| **Neural Autoencoder** | $0.07658$ | 0 | 16 | 83 | 1 | 0.00% | 0.0% | 0.0000 | 0.0370 | 0.7374 | 16.16% | **0.20 ms** |
| **Logistic Regression (Base)** | $0.50000$ | 0 | 9 | 90 | 1 | 0.00% | 0.0% | 0.0000 | 0.0323 | 0.6970 | 9.09% | 0.17 ms |
| **Logistic Regression (SMOTE)** | $0.50000$ | 0 | 7 | 92 | 1 | 0.00% | 0.0% | 0.0000 | 0.0278 | 0.6465 | 7.07% | 0.18 ms |

---

## ⚖️ Ethical Governance & DPDP Act 2023 Alignment
1. **PII Exclusion:** The system never asks for, processes, or persists Primary Account Numbers (PANs), CVVs, expiration dates, PINs, or customer names. All card accounts utilize masked tokens (`CARD_XXXX####`).
2. **Human-in-the-Loop Triage:** Anomaly flags provide risk signals for human compliance officers; no automated decline decisions occur.
3. **Anonymized Audit Trail:** `outputs/audit.db` stores only mathematical anomaly scores, transaction sequence numbers, model versions, and timestamps.
4. **Separated Synthetic Data:** The 4,500 synthetic demonstration transactions are isolated in `data/synthetic_transactions_only.csv` with a verified audit manifest (`synthetic_data_manifest.json`).

---

## 👨‍💻 Author and Academic Attestation
- **Learner:** Swapnil Bhardwaj
- **Roll Number:** [ENTER ROLL NUMBER]
- **Institution:** ABES Engineering College
- **Mentor:** [ENTER MENTOR NAME]
- **Application:** FraudGuard AI Enterprise Risk System (Digital-Data-AI / AIML)
