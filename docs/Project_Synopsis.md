# Project Synopsis — FraudGuard AI: Credit Card Fraud Detection Using Anomaly Detection
## Enterprise Application Specification | Advanced AI / AIML System | Academic Year 2025–26

| Field | Details |
|---|---|
| **Application Name** | FraudGuard AI — Enterprise Transaction Risk Intelligence |
| **Project Type** | Financial Crime & Anomaly Detection System |
| **Domain** | Digital-Data-AI / AIML |
| **Difficulty Level** | Advanced |
| **Duration** | 4 Weeks |
| **Format** | Individual Project |
| **Lead Developer** | Swapnil Bhardwaj |
| **Roll Number** | [ENTER ROLL NUMBER] |
| **College / Campus** | ABES Engineering College |
| **Mentor Name** | [ENTER MENTOR NAME] |
| **Submission Date** | [ENTER SUBMISSION DATE] |

---

### 1. Problem Statement
Financial institutions, payment processors, and cardholders face severe economic losses and operational disruption when unauthorized transactions evade payment authorization controls. The core technical hurdle is identifying subtle fraudulent spending anomalies amidst massive volumes of legitimate transaction activity characterized by extreme class imbalance ($1.2\%$ fraud in the provided dataset), without triggering excessive false alarms that disrupt legitimate commerce. This project designs, implements, and evaluates an unsupervised anomaly detection framework to prioritize high-risk transactions for human compliance review.

### 2. Proposed Solution & Approach
The solution is an end-to-end Python prototype featuring a 7-tab Streamlit dashboard, a FastAPI REST layer, and an anonymized SQLite audit logger. The system integrates five comparative modeling strategies: **Isolation Forest**, **Local Outlier Factor (LOF with `novelty=True`)**, a **Neural Autoencoder** utilizing hourglass bottleneck reconstruction error, a **Class-Weighted Logistic Regression Baseline**, and a **Supervised Baseline with within-fold SMOTE oversampling**. All preprocessing scalers and frequency encoders are fitted strictly on the training fold, while feature engineering prevents target leakage. Anomaly scores are presented as decision-support risk signals routing transactions to human reviewers rather than executing autonomous financial declines.

### 3. System Architecture
```
[ Incoming Transaction CSV / REST Payload ]
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 1. Ingestion & Validation               │ ──> Schema, Range, Datatype & Category Checks
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 2. Feature Engineering & Preprocessing  │ ──> Cyclical Hour/Day, Log1p Amount, Prior Velocity
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 3. Multi-Model Anomaly Scoring Engine   │ ──> Isolation Forest, LOF, Autoencoder, Supervised
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│ 4. Decision Support & Human Triage UI   │ ──> Streamlit UI, FastAPI REST API, SQLite Audit
└─────────────────────────────────────────┘
                    │
                    ▼
[ Human Compliance Officer Inspection — Zero Automated Declines ]
```

### 4. Technology Stack
- **Frontend / UI:** Streamlit 1.64.0 (7 interactive tabs: Overview, EDA, Training, Comparison, Single Scoring, Batch Export, Governance)
- **Backend / Logic:** Python 3.13.7; pandas 3.0.3; NumPy 2.5.1; FastAPI 0.162.0 REST API
- **Database / Audit:** SQLite (`outputs/audit.db`) with zero PII retention
- **AI/ML Frameworks:** scikit-learn 1.9.0 (Isolation Forest, LOF, MLP Autoencoder, Logistic Regression); imbalanced-learn 0.14.2 (SMOTE)
- **Visualisation:** Matplotlib 3.11.0; Plotly 7.1.0
- **Testing & QA:** pytest 9.1.1 (30 automated passing tests)

### 5. Dataset Used
- **Dataset File Name:** `dataset.csv` (original; standardized copy: `dataset_cleaned.csv`)
- **Number of Rows:** 500 records (494 legitimate, 6 fraud-labelled)
- **Number of Columns:** 8
- **Key Columns:** `trans_id`, `timestamp`, `card_token`, `amount`, `merchant_category`, `is_fraud`, `merchant_id`, `card_type`
- **Additional Data:** 4,500 synthetic demonstration records isolated in `synthetic_transactions_only.csv` with a separate audit manifest; original file unmodified.
- **Schema Discrepancy Note:** The supplied CSV has 8 specific business columns and is distinct from the public 284,807-row Kaggle ULB benchmark (`V1`–`V28`).

### 6. Key Results & Metrics
*(Empirical measurements from held-out evaluation set: 99 legitimate, 1 fraud)*
- **Metric 1: Fraud-class Precision:** $12.50\%$ ($1/8$ flagged) for Local Outlier Factor; $0.0\%$ for Isolation Forest at $2\%$ contamination threshold.
- **Metric 2: Fraud-class Recall:** $100.0\%$ ($1/1$ detected, $0$ missed) for Local Outlier Factor; $0.0\%$ ($0/1$ detected) for Isolation Forest.
- **Metric 3: F1 and Ranking Quality:** LOF achieved $\text{F1} = 0.2222$, $\text{PR-AUC} = 0.2000$, $\text{ROC-AUC} = 0.9596$. Isolation Forest achieved $\text{ROC-AUC} = 0.9899$ and $\text{PR-AUC} = 0.5000$.
- **Metric 4: Scoring Latency:** $0.17 \text{ ms}$ to $11.95 \text{ ms}$ per transaction record.
- **Business Value:** LOF automated triage filters $92.0\%$ of normal transaction volume while successfully catching the unseen fraud instance, routing only 8 items to human reviewers without disruption to legitimate customers.

### 7. Challenges Faced & How Resolved
- **Challenge 1: Extreme Minority Class Sparsity (only 6 fraud rows out of 500):**  
  *Resolution:* Applied stratified holdout ($80/20$), explicitly reported sample counts per split (Train: 395 legit, 5 fraud; Test: 99 legit, 1 fraud), stated wide confidence intervals, and added warnings against relying on accuracy.
- **Challenge 2: Schema Inconsistency across Documentation:**  
  *Resolution:* Standardized on the supplied 8-column CSV as ground truth and documented that the Kaggle ULB file represents a separate benchmark.
- **Challenge 3: Preventing Data Leakage in Feature Engineering & Oversampling:**  
  *Resolution:* Fitted all scalers and encoders on $X_{\text{train}}$ only; sorted data chronologically so card velocity counted strictly prior transactions; applied SMOTE strictly inside the training fold with dynamic $k_{\text{neighbors}} \le N_{\text{minority}} - 1$.

### 8. References
1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. IEEE ICDM, pp. 413–422.
2. Breunig, M. M., et al. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD, 29(2), pp. 93–104.
3. Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, pp. 321–357.
4. scikit-learn Documentation: IsolationForest & LocalOutlierFactor. [https://scikit-learn.org/](https://scikit-learn.org/)
5. Government of India: Digital Personal Data Protection Act, 2023. [https://egazette.gov.in/](https://egazette.gov.in/)

### 9. Ethical Compliance Declaration
I confirm that:
- [x] All dataset usage complies with the source licence and academic guidelines.
- [x] No personally identifiable information (PII) is stored; card tokens are masked and CVVs/PANs are excluded.
- [x] AI tools used for scaffolding and code development are disclosed in Appendix A.
- [x] This project complies with the DPDP Act 2023 through data minimization and human-in-the-loop triage.

**Learner Signature:** _________________________    **Date:** ________________
