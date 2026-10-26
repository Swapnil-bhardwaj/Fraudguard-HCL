# Credit Card Fraud Detection Using Anomaly Detection
## Submission starter pack — read this first

This pack is based on the supplied HCLTech Career Shaper Programme Project Brief, Synopsis Template, `dataset.csv`, `data_dictionary.csv`, and Dataset Generation Prompt screenshot.

## What is inside

- `prompts/MASTER_PROJECT_BUILD_PROMPT.txt` — the main prompt to paste into Cursor, Claude, Gemini, or another coding assistant to create the working application and report artifacts.
- `prompts/SYNTHETIC_DATA_GENERATION_PROMPT.txt` — optional prompt for generating extra records while keeping generated data clearly separated from the original.
- `docs/Project_Synopsis_Swapnil_Bhardwaj_ROLLNO_PENDING.docx` — synopsis drafted using the supplied template. Enter your roll number and mentor/date details; refresh metrics and challenges after running the project.
- `docs/Project_Report_Draft_Swapnil_Bhardwaj.docx` — an 8+ page report draft/structure. Evaluation claims are deliberately marked pending until models are actually trained and tested.
- `docs/Ethical_Compliance_Declaration.docx` — declaration template; review every item before checking/signing.
- `docs/system_architecture.png`, `docs/dfd_level_0.png`, `docs/architecture_combined.png` — diagrams for synopsis/report.
- `data/dataset.csv` — original 500 rows, retained as received; `data/dataset_cleaned.csv` — the original 500 records normalized to appropriate CSV formats. No rows were fabricated, added, or removed.
- `data/data_dictionary.csv` — completed field descriptions and expected analysis types; the original unfilled template is preserved as `data/data_dictionary_source_original.csv`.
- `src/clean_data.py` — reproducible CSV validation/cleaning script.
- `src/eda.py` — EDA script that creates four plots and an EDA summary.
- `outputs/eda_summary.json` and `outputs/figures/` — analysis of the original 500-row dataset.
- `submission_checklist.md` — deliverables checklist and four-week schedule.

## Facts from the supplied CSV

- 500 rows, 8 columns.
- `is_fraud`: 494 rows labelled 0 and 6 rows labelled 1 (1.2%).
- No missing values and no exact duplicate rows were found.
- `trans_id` is sequential from 1 to 500.
- `timestamp` increases from 2024-01-01 00:17:00 through 2024-01-11 19:58:00.
- Amount range: 3.52 to 4,241.75 (the CSV does not specify a currency).
- Card tokens are masked in the form `CARD_XXXX####`.

## Important dataset warning

The current CSV uses these columns: `trans_id, timestamp, card_token, amount, merchant_category, is_fraud, merchant_id, card_type`. It is **not** the public Kaggle ULB dataset described in the screenshot. The Kaggle ULB dataset normally uses `Time`, `V1`–`V28`, `Amount`, and `Class`, and contains 284,807 rows. Do not claim that the supplied file is the Kaggle dataset, and do not mix the two schemas. If the mentor requests 5,000 rows, generate additional rows as synthetic data and retain a separate copy/manifest; do not silently overwrite the original 500 rows.

## Recommended build approach

Use Python + Streamlit for the interface, pandas/NumPy for data handling, scikit-learn for Isolation Forest and LOF, TensorFlow/Keras for an optional autoencoder, imbalanced-learn for a supervised comparison using SMOTE/SMOTENC only on training data, and SQLite for minimal model-run/audit metadata. Use `is_fraud` only for evaluation/supervised training, never as a model input. Evaluate fraud precision, recall, F1, Average Precision/PR-AUC, confusion matrix, false-positive rate, and prediction latency; do not present accuracy alone.

## Before submission

1. Fill your roll number, mentor, date, and signature.
2. Run the models and replace every `PENDING` metric with measured output.
3. Document the exact data source, generated-data method (if any), software versions, random seeds, train/validation/test strategy, and the AI tools actually used.
4. Verify the app from a fresh environment and record a 1–2 minute demo.
5. Ask your mentor to confirm whether synthetic augmentation is allowed and whether the supplied CSV or Kaggle ULB file is the required dataset.
