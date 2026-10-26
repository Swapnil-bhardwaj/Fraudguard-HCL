# Final Submission Checklist

## Documents and evidence required by the supplied brief

- [ ] Working fraud-detection prototype/application.
- [ ] Project report of at least 8 pages, completed using actual run results.
- [ ] EDA notebook or analysis script with at least 3 visualisations (`src/eda.py` supplies four reproducible visualisations).
- [ ] Cleaned/preprocessed CSV derived from the supplied data (`data/dataset_cleaned.csv`).
- [ ] Completed data dictionary (`data/data_dictionary_completed.csv`).
- [ ] 1–2 minute demo video or a live demo.
- [ ] Ethical compliance declaration reviewed and signed.
- [ ] Synopsis based on the supplied programme template.
- [ ] Appendix disclosing actual AI tools used.

## Data validation before running models

- [ ] Confirm the mentor accepts the provided eight-column schema.
- [ ] Preserve the original CSV unmodified.
- [ ] If adding generated rows, save them separately and document that they are synthetic.
- [ ] Confirm the label coding (`0` = legitimate, `1` = fraud).
- [ ] Verify no `is_fraud` leakage into model features.
- [ ] Fit all encoders/scalers/oversamplers on training data only.
- [ ] Skip or adapt SMOTE if the training set does not contain enough minority samples; do not silently force it.

## Evaluation checklist

- [ ] State the split strategy and the number of fraud examples in train, validation, and test sets.
- [ ] Record precision, recall, F1, Average Precision/PR-AUC, confusion matrix, false-positive rate, and inference latency.
- [ ] Compare Isolation Forest, LOF, and Autoencoder where feasible; add a supervised baseline with SMOTE/SMOTENC only if the available labels support it.
- [ ] State clearly when a score is based on synthetic data and avoid claims of real-world performance.
- [ ] Save model configuration, thresholds, random seed, dependency versions, and output metrics.
- [ ] Include limitations and false-positive / false-negative discussion.

## Suggested four-week timeline

| Week | Work | Checkpoint |
|---|---|---|
| 1 | Understand brief, validate data/schema, EDA and visualisations | Problem statement and EDA reviewed by mentor |
| 2 | Architecture, preprocessing, anomaly models, first interface | Working module demo |
| 3 | End-to-end integration, validation, thresholding, tests, metrics | Repeatable results and measured performance |
| 4 | Finish report/synopsis, ethical review, demo recording, presentation | Complete submission and expo demo |
