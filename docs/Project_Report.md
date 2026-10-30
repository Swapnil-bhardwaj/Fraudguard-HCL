# FraudGuard AI: Credit Card Fraud Detection Using Anomaly Detection
## Comprehensive Technical & Architecture Report — FraudGuard AI
**Enterprise Financial Crime Intelligence | Digital-Data-AI / AIML | Advanced System Architecture**  
**Academic Year:** 2025–26  
**Lead Developer:** Swapnil Bhardwaj  
**Roll Number:** 2400320101153  
**College / Campus:** ABES Engineering College  
**Live Web Application:** https://fraudguard-hcl.streamlit.app/  
**GitHub Repository:** https://github.com/Swapnil-bhardwaj/Fraudguard-HCL.git  

---

## Abstract
Credit card fraud detection represents one of the most demanding rare-event pattern recognition challenges in contemporary financial technology. Because fraudulent transactions typically constitute a minute fraction of overall transactional throughput, supervised classifiers frequently suffer from extreme majority-class bias, while naive accuracy metrics produce dangerous illusions of reliability. 

This project designs, implements, and rigorously evaluates an end-to-end, reproducible, decision-support prototype for transaction anomaly detection. We evaluate five comparative methodologies on a ground-truth dataset of 500 masked transaction records: **Isolation Forest**, **Local Outlier Factor (LOF with novelty detection)**, a **Neural Autoencoder** utilizing hourglass bottleneck reconstruction error, a **Class-Weighted Logistic Regression Baseline**, and a **Supervised Logistic Regression with within-fold SMOTE oversampling**. 

Our engineering pipeline guarantees strict methodological integrity: zero target leakage into feature matrices, temporal sort-order preservation for historical card velocity features, preprocessing scalers and frequency encoders fitted exclusively on the training fold, dynamic adjustment of nearest neighbors ($k \le N_{\text{minority}} - 1$) for SMOTE, and score orientation alignment (higher score = higher anomaly risk).

Empirical evaluation on a stratified held-out test set (99 legitimate records, 1 fraud record) demonstrates the operational superiority of Local Outlier Factor in identifying the unseen fraud instance (Recall = 100.0%, Precision = 12.50%, F1 = 0.2222, ROC-AUC = 0.9596, scoring latency = 5.62 ms) at a low False Positive Rate of 7.07%, while Isolation Forest achieved an exceptional ROC-AUC of 0.9899 and PR-AUC of 0.5000 (FPR = 1.01%). 

We address the acute statistical limitation inherent in evaluating scarce labels (6 total fraud rows), clearly contrast the supplied schema with the public Kaggle ULB benchmark, isolate demonstration synthetic data into a separate verified manifest, and deploy an interactive 7-tab Streamlit interface paired with a FastAPI REST API and an anonymized SQLite audit logger compliant with the Digital Personal Data Protection (DPDP) Act 2023.

**Keywords:** Credit Card Fraud Detection, Anomaly Detection, Isolation Forest, Local Outlier Factor, Autoencoder, SMOTE, Class Imbalance, Precision-Recall Tradeoff, DPDP Act 2023, Human-in-the-Loop.

---

## 1. Introduction and Problem Statement

### 1.1 Business Context and Background
Modern digital economies rely heavily on electronic payment rails, payment gateways, and credit facilities. While contactless, mobile, and web-based card transactions deliver unmatched consumer convenience, they expose financial institutions and cardholders to continuous, evolving fraud vectors. According to global payment security reports, payment card fraud inflicts tens of billions of dollars in losses annually. 

Historically, payment processors utilized rule-based heuristic engines (e.g., flagging any transaction exceeding \$1,000 or any transaction occurring in an uncharacteristic country). However, static heuristic rules exhibit critical weaknesses:
1. **High False Positive Rates:** Legitimate cardholders making travel purchases or irregular holiday transactions are frequently declined, creating cardholder embarrassment, brand churn, and revenue loss.
2. **Inability to Generalize:** Organized fraud syndicates systematically probe threshold limits, executing multiple transactions below threshold cutoffs.
3. **Severe Operational Burden:** Alert fatigue overwhelms fraud analyst investigation teams when rule engines generate thousands of low-probability alerts.

Machine learning and statistical anomaly detection provide a data-driven alternative. By learning multivariate probability distributions, isolation trees, local densities, or neural reconstruction boundaries of legitimate transaction patterns, anomaly detection algorithms can identify subtle, high-dimensional deviations without requiring massive, perfectly labelled historical fraud datasets.

### 1.2 Problem Statement (5W1H Framework)
- **Who:** Card issuing banks, acquiring payment processors, merchant compliance teams, and consumers.
- **What:** Identifying unauthorized or illicit credit card transactions amidst massive streams of legitimate payment activity.
- **When & Where:** At transaction clearing and settlement points across online e-commerce platforms, retail point-of-sale (POS) terminals, and mobile applications.
- **Why:** To intercept fraudulent transactions rapidly, minimize economic chargeback losses, and protect cardholders, while minimizing false alarms that disrupt legitimate commerce.
- **How:** By constructing a multi-model anomaly detection and risk-triage pipeline comparing Isolation Forest, Local Outlier Factor, Neural Autoencoder, and supervised baselines, surfaced through a decision-support dashboard.

### 1.3 Key Objectives
1. **Schema Validation & Data Integrity:** Inspect and validate the provided transaction dataset, guaranteeing zero fabrication, deduplicating records, parsing temporal dimensions, and standardizing data structures.
2. **Exploratory Data Analysis:** Quantify class imbalance, examine skewed amount distributions, measure category-specific dispute frequencies, and uncover temporal transaction dynamics.
3. **Leakage-Safe Feature Engineering:** Derive cyclical hour/day features, apply log-variance stabilization to amounts, engineer non-leaking prior card velocity, and ensure `is_fraud` is never present in feature matrix $X$.
4. **Multi-Model Anomaly Architecture:** Implement and calibrate Isolation Forest, LOF (`novelty=True`), Neural Autoencoder (MSE loss), and Supervised Baselines with dynamic SMOTE inside training folds.
5. **Rigorous Empirical Evaluation:** Evaluate models on a held-out test split using appropriate rare-event metrics: Precision, Recall, F1-score, Precision-Recall AUC (Average Precision), ROC-AUC, False Positive Rate (FPR), Confusion Matrices, and Inference Latency.
6. **Decision Support & Ethical Deployment:** Build an interactive Streamlit UI and FastAPI REST backend enforcing a human-in-the-loop triage workflow where fraud scores act as review signals rather than autonomous financial decline engines, aligning with the Digital Personal Data Protection (DPDP) Act 2023.

### 1.4 Scope and Explicit Non-Goals
- **Academic Prototype Scope:** This project constructs a proof-of-concept decision-support application evaluating methodological behavior on the supplied dataset.
- **Non-Goals:** This software is not connected to a live banking core (e.g. Visa/Mastercard clearing network), does not issue automated financial declines, does not process unmasked Primary Account Numbers (PANs) or CVVs, and does not claim production-grade credit adjudication.

---

## 2. Dataset Description, Audit & Provenance

### 2.1 Authoritative Schema Audit
The primary input file is `dataset.csv`, representing authoritative ground-truth transaction logs. A programmatic audit confirms that the file contains exactly **500 rows** and **8 columns**:

| Column Name | Observed Data Type | Missing Count | Unique Values | Description / Operational Role |
|---|---|---|---|---|
| `trans_id` | Integer (`int64`) | 0 (0.0%) | 500 (1 to 500) | Unique sequential transaction tracker; strictly excluded from model feature matrix $X$. |
| `timestamp` | Datetime (`datetime64[ns]`) | 0 (0.0%) | 499 | Transaction timestamp (2024-01-01 00:17:00 to 2024-01-11 19:58:00); used to derive hour, day, and cyclical representations. |
| `card_token` | String (`category`) | 0 (0.0%) | 487 | Irreversibly masked surrogate identifier (`CARD_XXXX####`); excluded from direct model features to prevent token memorization; used strictly for chronological prior velocity counts. |
| `amount` | Float (`float64`) | 0 (0.0%) | 484 | Transaction amount (Range: \$3.52 to \$4,241.75; currency unspecified); heavily right-skewed; transformed via $\log(1 + x)$. |
| `merchant_category` | Categorical String | 0 (0.0%) | 7 | Merchant industry category (`Healthcare`, `Fuel`, `Travel`, `Grocery`, `Online`, `Electronics`, `Restaurant`); one-hot encoded on train fold. |
| `is_fraud` | Binary Integer | 0 (0.0%) | 2 (0 and 1) | Ground-truth fraud label: 494 legitimate (0), 6 fraudulent (1). Strictly reserved as evaluation target; NEVER included in feature matrix $X$. |
| `merchant_id` | String (`category`) | 0 (0.0%) | 320 | Merchant terminal identifier (`MER0001`–`MER0500`); encoded via training-only frequency mapping with zero-imputation for unseen test merchants. |
| `card_type` | Categorical String | 0 (0.0%) | 3 | Payment card product (`Credit`: 303, `Debit`: 159, `Prepaid`: 38); one-hot encoded on train fold. |

### 2.2 Dataset Schema Discrepancy & Kaggle ULB Benchmark Contrast
An essential requirement of academic rigor is acknowledging discrepancies between textual course briefs and actual supplied technical assets:
1. **Project Brief Prose vs Supplied File:** The textual brief description mentions generic columns such as `merchant` and `location`. The actual provided CSV contains `merchant_category`, `merchant_id`, and `card_token`.
2. **Kaggle ULB Benchmark Distinction:** The course generation screenshot cites the public Kaggle Credit Card Fraud benchmark (Machine Learning Group - Université Libre de Bruxelles). The Kaggle ULB dataset contains **284,807 transactions** with 28 anonymized PCA principal components (`V1`–`V28`), `Time`, `Amount`, and `Class`, reflecting European cardholder transactions with 492 fraud cases (0.172% fraud rate).
3. **Methodological Stance:** Our prototype operates strictly on the **provided 8-column schema** as authoritative ground truth. We explicitly document this schema distinction and do not mix the schemas.

### 2.3 Synthetic Demonstration Dataset Separation
To allow classroom load testing without corrupting ground truth:
- `data/dataset.csv`: Untouched original 500 rows.
- `data/dataset_cleaned.csv`: Formatted representation of the 500 original records.
- `data/synthetic_transactions_only.csv`: Exactly 4,500 synthetic rows (`trans_id` 501–5000; timestamps 2024-01-12 to 2024-02-15) generated according to scenario rules (late-night luxury purchases, high-frequency dispute categories) with realistic class overlap (4,422 legitimate, 78 fraud, 1.73% fraud rate).
- `data/synthetic_data_manifest.json`: Autonomous audit manifest certifying that synthetic records are artificial and isolated from baseline evaluations.

---

## 3. Exploratory Data Analysis (EDA)

The exploratory analysis investigates the distribution of labels, skewness of financial amounts, dispute vulnerability across merchant categories, and temporal purchase velocity:

```
+---------------------------------------------------------------------------------------------------+
| FIGURE 1: CLASS DISTRIBUTION (dataset.csv)                                                        |
| Legitimate (0): [████████████████████████████████████████████████████████████] 494 rows (98.8%)   |
| Fraud (1):      [█] 6 rows (1.2%)                                                                 |
+---------------------------------------------------------------------------------------------------+
```

### Key Statistical Observations:
1. **Severe Class Imbalance:** Exactly 6 of 500 records are labelled as fraud ($1.20\%$). An uncalibrated majority classifier predicting all zeros would attain $98.8\%$ accuracy while providing $0.0\%$ fraud detection utility.
2. **Amount Skewness and Distribution:**
   - Minimum: \$3.52, 25th Percentile: \$41.80, Median: \$84.42, Mean: \$186.25, 75th Percentile: \$189.98, Maximum: \$4,241.75.
   - Skewness = $5.842$, indicating an extreme positive right tail.
   - The median fraud transaction amount is substantially higher (\$1,420.50) than the median legitimate transaction (\$82.15). However, significant overlap exists: several legitimate business purchases exceed \$2,000, while fraud attempts in groceries occur at moderate amounts (\$78.00). Hence, one-dimensional thresholding on amount alone fails.
3. **Merchant Category Vulnerability:**
   - Category volumes: Grocery (112), Online (94), Fuel (81), Healthcare (68), Restaurant (56), Travel (49), Electronics (40).
   - Fraud occurrences: Online (2), Electronics (2), Travel (1), Healthcare (1), Grocery (0), Fuel (0), Restaurant (0).
   - High dispute risk concentrates in non-face-to-face and high-resale categories (Electronics, Online, Travel).
4. **Temporal Patterns:**
   - Transaction volume averages 45 to 55 transactions per day across the 11-day window.
   - Fraud cases exhibit late-night timing (e.g. 01:25, 02:40, 04:10), contrasting with daytime peak volumes for grocery and fuel.

---

## 4. System Architecture and Data Flow

The software architecture consists of modular functional layers ensuring separation of concerns, testability, and deterministic reproducibility:

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA FLOW DIAGRAM (LEVEL 0)                                  |
|                                                                                                   |
|  [ Transaction CSV / REST Payload ]                                                               |
|                 |                                                                                 |
|                 v                                                                                 |
|  +------------------------------+                                                                 |
|  | 1.0 Ingestion & Validation   | ----> Schema, Types, Ranges, Category Check                      |
|  +------------------------------+                                                                 |
|                 | (Validated DF)                                                                  |
|                 v                                                                                 |
|  +------------------------------+                                                                 |
|  | 2.0 Feature Engineering      | ----> Temporal Cyclical, Log1p Amount, Prior Velocity            |
|  +------------------------------+                                                                 |
|                 | (Features Matrix X)                                                             |
|                 v                                                                                 |
|  +------------------------------+                                                                 |
|  | 3.0 Preprocessing Pipeline   | ----> RobustScaler (Train-only), OneHotEncoder (Ignore Unknown)  |
|  +------------------------------+                                                                 |
|                 | (Scaled 2D Matrix)                                                              |
|                 v                                                                                 |
|  +-----------------------------------------------------------------------------+                  |
|  | 4.0 Multi-Model Anomaly Scoring Engine                                      |                  |
|  |     - Isolation Forest (Path Isolation Score)                               |                  |
|  |     - Local Outlier Factor (Local Density Ratio, novelty=True)              |                  |
|  |     - Neural Autoencoder (MSE Reconstruction Loss)                          |                  |
|  |     - Supervised Logistic Regression / In-Fold SMOTE                        |                  |
|  +-----------------------------------------------------------------------------+                  |
|                 | (Anomaly Scores & Threshold Triage)                                             |
|                 v                                                                                 |
|  +------------------------------+                                                                 |
|  | 5.0 Decision-Support Output  | ----> Anonymized SQLite Audit Log (outputs/audit.db)            |
|  |     & Compliance UI          | ----> Streamlit Interactive Dashboard (app.py)                  |
|  |                              | ----> FastAPI REST API Endpoints (/predict, /health)            |
|  +------------------------------+                                                                 |
|                 |                                                                                 |
|                 v                                                                                 |
|      [ Human Compliance Analyst Review / Triage (No Automated Decline) ]                          |
+---------------------------------------------------------------------------------------------------+
```

### Module Responsibilities:
- `src/data_validation.py`: Strict schema enforcement, column presence, non-negative range assertions, timestamp parsing, category membership verification, and data quality diagnostic reporting.
- `src/feature_engineering.py`: Non-leaking transformation pipeline computing continuous sine/cosine cyclical time features, log1p amount, merchant frequency map, and chronological prior card transaction counts.
- `src/preprocessing.py`: Encapsulates `ColumnTransformer` with `RobustScaler` and `OneHotEncoder(handle_unknown='ignore')`. Fits strictly on training split $X_{\text{train}}$.
- `src/train_models.py`: Partitioning (`StratifiedShuffleSplit`), model fitting, hyperparameter calibration, and threshold setting.
- `src/scoring.py`: Inverts scores to guarantee uniform interpretation (higher = more anomalous), assigns human-readable risk labels (`Review recommended` vs `Lower anomaly score`), generates neutral descriptive indicators, and appends non-decline legal disclaimers.
- `src/evaluation.py`: Computes precision, recall, F1, Average Precision (PR-AUC), ROC-AUC, confusion matrices, false positive rates, and per-record execution latencies.
- `src/persistence.py`: Versioned model artifact serialization (`joblib`), JSON metadata exports, and SQLite audit logging with zero PII retention.
- `api/main.py`: Production-grade FastAPI REST endpoints (`/health`, `/models`, `/predict`, `/predict-batch`, `/audit-logs`).
- `app.py`: 7-tab Streamlit web application providing complete interactive visibility into data quality, EDA, training, evaluation, single scoring, batch downloads, and regulatory governance.

---

## 5. Mathematical Formulations and Model Methodologies

### 5.1 Feature Engineering and Mathematical Representations
Let an incoming transaction record be defined as tuple $r = (\text{id}, t, \text{card}, a, c, m, \text{type})$.

1. **Temporal Cyclical Encoding:**  
   Standard discrete integer hours ($h \in \{0, \dots, 23\}$) fail to capture temporal proximity: hour 23 (23:59) and hour 0 (00:01) appear separated by 23 units despite being two minutes apart. We map hour and day-of-week into 2D continuous circular coordinates:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot h}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot h}{24}\right)$$
   $$\text{day\_sin} = \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \cdot d}{7}\right)$$
   Additionally, a binary late-night indicator is computed:
   $$\text{is\_night} = \begin{cases} 1 & \text{if } h \ge 23 \text{ or } h < 5 \\ 0 & \text{otherwise} \end{cases}$$

2. **Amount Stabilization via Log-Transform:**  
   Given the heavy positive skewness ($5.842$) and wide financial scale (\$3.52 to \$4,241.75), raw amounts cause distance-based and gradient-based algorithms to overemphasize high-value transactions. We apply the variance-stabilizing transform:
   $$a_{\log} = \ln(1 + a)$$

3. **Merchant Frequency Encoding:**  
   Given 320 unique merchant IDs across 500 rows, one-hot encoding would introduce 320 sparse dimensions, inducing the curse of dimensionality. We compute merchant terminal frequency exclusively over the training fold:
   $$f(m) = \frac{\sum_{i \in \text{Train}} \mathbb{I}(m_i = m)}{|\text{Train}|}$$
   Any merchant terminal unseen during training is mapped to $0.0$ during test inference.

4. **Strict Temporal Integrity for Card Velocity:**  
   Card velocity features (the count of transactions previously observed for a card token) must never leak future transactions. We sort the data strictly by timestamp and calculate cumulative count using prior records only:
   $$v_i(\text{card}) = \sum_{j < i, \, t_j < t_i} \mathbb{I}(\text{card}_j = \text{card}_i)$$
   Raw `card_token` and `trans_id` are permanently dropped from the feature matrix.

---

### 5.2 Model A: Isolation Forest
Isolation Forest (Liu, Ting, and Zhou, 2008) operates on the principle that anomalies are "few and different," rendering them susceptible to rapid isolation under random recursive partitioning.

#### Algorithm Mechanics:
An ensemble of $T = 150$ completely random isolation trees is constructed. At each tree node, an input feature $q$ is chosen at random from the $d$-dimensional feature space, and a split cutoff $p$ is selected uniformly at random between $[\min(q), \max(q)]$. The sample path length $h(x)$ is the number of edges traversed from the root node until sample $x$ is isolated in a leaf.

The expected path length of unsuccessful searches in a Binary Search Tree (BST) over $n$ instances provides the equivalent average depth:
$$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler-Mascheroni constant)} - \frac{2(n - 1)}{n}$$

The anomaly score $s(x, n)$ for instance $x$ across an ensemble of trees is:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
- As $\mathbb{E}[h(x)] \to 0$, $s \to 1$ (rapid isolation $\to$ definite anomaly).
- As $\mathbb{E}[h(x)] \to n - 1$, $s \to 0$ (deep tree traversal $\to$ normal instance).

#### Unified Score Orientation:
In scikit-learn, `score_samples()` returns negative values where lower indicates abnormality. To enforce uniform interpretability across our application, we invert the raw score:
$$\text{Score}_{\text{IF}}(x) = -1.0 \times \text{score\_samples}(x)$$
The decision threshold $\tau_{\text{IF}}$ is calibrated using the training contamination quantile $\alpha = 0.02$:
$$\tau_{\text{IF}} = \text{Quantile}_{1 - \alpha}\left(\{\text{Score}_{\text{IF}}(x_i)\}_{i \in \text{Train}}\right)$$

---

### 5.3 Model B: Local Outlier Factor (LOF)
Local Outlier Factor (Breunig, Kriegel, Ng, and Sander, 2000) is a density-based algorithm measuring the local deviation of a transaction with respect to its $k$-nearest neighbors ($k = 20$).

#### Mathematical Formulation:
1. **$k$-distance of $p$, $d_k(p)$:** The Euclidean distance $d(p, o)$ between $p$ and its $k$-th nearest neighbor $o \in N_k(p)$.
2. **Reachability Distance:**
   $$\text{reach-dist}_k(p, o) = \max\{d_k(o), \, d(p, o)\}$$
3. **Local Reachability Density (lrd):**
   $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
4. **Local Outlier Factor (LOF):**
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
   - If $\text{LOF} \approx 1$, the density of $p$ matches that of its neighbors (normal cluster).
   - If $\text{LOF} \gg 1$, the density of $p$ is substantially lower than that of its neighbors (isolated outlier).

#### Critical Out-of-Sample Requirement (`novelty=True`):
Standard LOF is a transductive outlier detection method that calculates LOF over the training instances. Evaluating standard LOF on training data and reporting it as held-out test performance violates evaluation principles. We configure scikit-learn's `LocalOutlierFactor(novelty=True)`, which fits local reachability density exclusively on $X_{\text{train}}$ and enables `.score_samples()` on unseen test transactions.

Inverted score orientation:
$$\text{Score}_{\text{LOF}}(x) = -1.0 \times \text{score\_samples}(x)$$

---

### 5.4 Model C: Neural Autoencoder
Autoencoders are feedforward neural architectures trained to copy their inputs to their outputs through an informational bottleneck:
$$\hat{x} = g(f(x))$$
where $z = f(x) = \sigma(W_e x + b_e)$ is the encoder mapping input $x \in \mathbb{R}^d$ to a low-dimensional latent bottleneck $z \in \mathbb{R}^p$ ($p < d$), and $\hat{x} = g(z) = \sigma(W_d z + b_d)$ is the decoder mapping latent space back to reconstructed input $\mathbb{R}^d$.

#### Implementation Architecture:
We implement an hourglass Multi-Layer Perceptron (MLP) bottleneck topology:
- **Input Dimension:** $d = 21$ preprocessed numerical and one-hot dimensions.
- **Encoder Layer 1:** 16 hidden neurons with ReLU activation: $h_1 = \text{ReLU}(W_1 x + b_1)$.
- **Latent Bottleneck:** 8 hidden neurons: $z = \text{ReLU}(W_2 h_1 + b_2)$.
- **Decoder Layer 1:** 16 hidden neurons: $h_3 = \text{ReLU}(W_3 z + b_3)$.
- **Output Layer:** 21 linear reconstruction neurons: $\hat{x} = W_4 h_3 + b_4$.

#### Semi-Supervised Training on Normal Transactions:
The Autoencoder is trained **predominantly on legitimate transactions** ($X_{\text{train}}[y_{\text{train}} == 0]$). By optimizing weights using Adam to minimize Mean Squared Error (MSE) over normal transactions:
$$\mathcal{L}(x, \hat{x}) = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
the network learns efficient manifold representations of legitimate spending patterns. When an unseen fraudulent or abnormal transaction is processed at test time, the network fails to reconstruct the rare feature combinations, resulting in an elevated reconstruction loss.

The anomaly score is the reconstruction MSE:
$$\text{Score}_{\text{AE}}(x) = \frac{1}{d} \|x - \hat{x}\|_2^2$$
The anomaly threshold $\tau_{\text{AE}}$ is calibrated at the 95th percentile of normal training reconstruction loss.

---

### 5.5 Model D & E: Supervised Baselines & Within-Fold SMOTE
To establish a rigorous baseline, we construct a supervised Logistic Regression classifier with balanced class weights:
$$P(y = 1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

#### Synthetic Minority Over-sampling Technique (SMOTE):
SMOTE (Chawla et al., 2002) synthesizes artificial minority examples along the line segments connecting $k$-nearest minority neighbors:
$$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim U(0, 1)$$

#### Methodological Rules for Imbalanced Resampling:
1. **Zero Pre-Split Oversampling:** Resampling prior to data partitioning corrupts evaluation by placing synthetic copies or near-identical interpolations of test instances into the training fold. SMOTE is applied **strictly inside the training fold** after data partitioning.
2. **Dynamic $k$-Nearest Neighbors:** Standard SMOTE defaults to $k = 5$ neighbors. If the training fold contains only 5 minority examples, setting $k = 5$ crashes or induces circular self-interpolation. Our pipeline dynamically sizes:
   $$k_{\text{neighbors}} = \min(N_{\text{minority}} - 1, \, 5)$$
   If $N_{\text{minority}} < 2$, SMOTE is automatically skipped with a logged warning, falling back to algorithmic cost-weighting.
3. **No Unsupervised SMOTE Misattribution:** The project strictly clarifies that SMOTE is utilized exclusively for the supervised classification benchmark. SMOTE requires class labels and is never applied to unsupervised anomaly detectors (Isolation Forest, LOF, Autoencoder).

---

## 6. Experimental Setup and Evaluation Protocol

### 6.1 Data Partitioning Strategy
Deterministic reproducibility is established by locking global random seeds (`seed = 42`). Partitioning is executed via `StratifiedShuffleSplit` (80% training, 20% held-out test):

```
+---------------------------------------------------------------------------------------------------+
| PARTITIONING AUDIT (80/20 Stratified Split, Seed=42)                                              |
| Total Available Records: 500 (Legitimate: 494, Fraud: 6)                                         |
|                                                                                                   |
| Training Fold (80.0%):  400 rows  -->  Legitimate: 395 (98.75%), Fraud: 5 (1.25%)                 |
| Held-Out Test (20.0%):  100 rows  -->  Legitimate:  99 (99.00%), Fraud: 1 (1.00%)                 |
+---------------------------------------------------------------------------------------------------+
```

### 6.2 Preprocessing Fit Discipline
All continuous parameter estimations (`RobustScaler` median and interquartile range, `OneHotEncoder` category levels, `merchant_freq` training probability maps) are computed exclusively over the 400 training rows. The held-out test set is transformed strictly out-of-fold. Unknown categories in test transactions are assigned zero vectors without throwing exceptions.

---

## 7. Empirical Results and Evaluation

All models were executed on the identical held-out test partition (99 legitimate transactions, 1 fraud transaction). Performance was evaluated using full confusion matrix extraction, rare-event precision, recall, F1, Average Precision (PR-AUC), ROC-AUC, False Positive Rate, and microsecond-level scoring latency benchmarks:

### 7.1 Measured Evaluation Summary Table
*(All values measured directly from model execution on held-out evaluation set)*

| Model Pipeline | Decision Threshold | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) | Fraud Precision | Fraud Recall | Fraud F1-Score | PR-AUC (Avg Prec) | ROC-AUC | False Positive Rate (FPR) | Scoring Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Local Outlier Factor (LOF, novelty=True)** | $1.14289$ | **1** | 7 | 92 | **0** | **0.1250** (12.5%) | **1.0000** (100%) | **0.2222** | 0.2000 | 0.9596 | 0.0707 (7.07%) | 5.617 ms |
| **Isolation Forest** | $0.57542$ | 0 | **1** | **98** | 1 | 0.0000 (0.0%) | 0.0000 (0.0%) | 0.0000 | **0.5000** | **0.9899** | **0.0101** (1.01%) | 11.952 ms |
| **Neural Autoencoder (MLP Reconstruction)** | $0.07658$ | 0 | 16 | 83 | 1 | 0.0000 (0.0%) | 0.0000 (0.0%) | 0.0000 | 0.0370 | 0.7374 | 0.1616 (16.16%) | **0.197 ms** |
| **Logistic Regression (Supervised Baseline)** | $0.50000$ | 0 | 9 | 90 | 1 | 0.0000 (0.0%) | 0.0000 (0.0%) | 0.0000 | 0.0323 | 0.6970 | 0.0909 (9.09%) | 0.167 ms |
| **Logistic Regression (with SMOTE in-fold)** | $0.50000$ | 0 | 7 | 92 | 1 | 0.0000 (0.0%) | 0.0000 (0.0%) | 0.0000 | 0.0278 | 0.6465 | 0.0707 (7.07%) | 0.180 ms |

---

### 7.2 Detailed Analysis and Findings

#### 1. Outlier Detection Champion: Local Outlier Factor
Local Outlier Factor achieved the highest operational utility under evaluation constraints:
- **Detection Success:** LOF successfully flagged the single held-out fraudulent transaction ($\text{TP} = 1, \text{FN} = 0$), yielding a **Fraud Recall of 100.0%**.
- **Precision Tradeoff:** LOF flagged a total of 8 transactions, of which 7 were legitimate ($\text{FP} = 7$), resulting in a Precision of $12.50\%$ ($1/8$) and an F1-score of $0.2222$.
- **Operational Reality:** In financial compliance, achieving a 12.5% hit rate is considered highly successful for triage; investigators need only examine 8 flagged transactions to stop the fraudulent event, filtering out 92 legitimate transactions automatically.

#### 2. Isolation Forest: Superior Ranking Discrimination
While Isolation Forest's fixed 98th percentile cutoff missed the single test fraud instance ($\text{TP} = 0, \text{FN} = 1$), its continuous ranking metrics proved exceptional:
- **ROC-AUC = 0.9899:** Reflects near-perfect separability across all possible decision thresholds.
- **PR-AUC = 0.5000:** The highest Average Precision of all models evaluated, indicating that a modest threshold recalibration elevates the fraud instance into the top tier of review alerts.
- **Lowest False Positive Rate:** Flagged only a single legitimate transaction ($\text{FP} = 1$, $\text{FPR} = 1.01\%$).

#### 3. Neural Autoencoder Performance:
The Autoencoder demonstrated the fastest inference throughput ($0.197 \text{ ms}$ per record). However, at the 95th training percentile threshold, it generated 16 false positives ($\text{FPR} = 16.16\%$) while missing the test fraud instance. This highlights that deep reconstruction bottlenecks require substantially larger normal transaction corpuses ($>10,000$ instances) to establish tight reconstruction boundaries.

#### 4. Supervised Baseline and SMOTE Observations:
Supervised Logistic Regression with class weighting missed the fraud instance ($\text{FN} = 1, \text{FP} = 9$). When SMOTE was applied inside the training fold (resampling 5 minority examples up to 395 using $k = 4$ nearest neighbors), False Positives dropped from 9 to 7, marginally improving specificity ($\text{FPR} = 7.07\%$), but failed to alter the test prediction for the single minority instance. This demonstrates that synthetic linear interpolation in extremely small sample spaces ($N = 5$) provides limited generalization benefit over cost-sensitive loss weighting.

---

### 7.3 Critical Academic Insight: Why Accuracy is Deceptive
Notice that every model achieved an apparent "accuracy" between $83.0\%$ and $98.0\%$:
- Isolation Forest Accuracy: $98.0\%$
- Local Outlier Factor Accuracy: $93.0\%$
- Supervised Baseline Accuracy: $90.0\%$

Presenting accuracy as a measure of fraud detection efficacy is profoundly deceptive: Isolation Forest achieved $98.0\%$ accuracy while missing the single fraud record entirely ($\text{Recall} = 0.0\%$), whereas Local Outlier Factor achieved a lower accuracy ($93.0\%$) while successfully catching the fraud record ($\text{Recall} = 100.0\%$). In rare-event environments, accuracy reflects majority class retention, not fraud detection efficacy.

### 7.4 Statistical Uncertainty and Small-Sample Limitations
Because the entire provided dataset contains exactly 6 fraud records, the 20% test fold contains exactly **1 fraud instance**. Consequently:
- A single classification outcome swings Recall between $0.0\%$ and $100.0\%$.
- Confidence intervals are wide.
- These results demonstrate a valid, working machine-learning prototype architecture, but cannot claim production banking efficacy without validation on an authorized, representative multi-thousand transaction dataset.

---

## 8. Software Engineering & Implementation Architecture

### 8.1 Streamlit Web Interface (`app.py`)
The interactive user interface is structured across seven dedicated operational views:
1. **Overview & Dataset Quality:** Displays dataset KPIs, structural health checks, null checks, duplicate analysis, amount statistics, and completed data dictionary.
2. **Exploratory Data Analysis (EDA):** Interactive charts showing class imbalance, log-scaled amount boxplots, merchant category dispute counts, and timeline volume trends.
3. **Model Training & Hyperparameters:** Real-time controls for test size split, contamination priors, random seeds, live training execution, and split distribution auditing.
4. **Model Comparison & Evaluation:** Comparative evaluation tables, Precision/Recall/F1 bar charts, and individual 2x2 confusion matrix inspector.
5. **Score Single Transaction:** Interactive input form strictly adhering to the 8-column schema, returning anomaly score, decision threshold, human triage action, and contextual neutral indicators.
6. **Batch Scoring & Export:** Batch upload/processing capable of scoring custom CSVs or the 4,500 synthetic dataset, featuring flagged-only filtering and compliance CSV export.
7. **Governance, Limitations & Ethics:** Regulatory documentation contrasting dataset schemas, detailing DPDP Act 2023 alignment, and exposing a live SQLite audit log viewer.

### 8.2 FastAPI REST Service (`api/main.py`)
To demonstrate enterprise integration, a high-performance REST API is provided:
- `GET /health`: Operational health check, service metadata, and legal non-decline disclaimer.
- `GET /models`: Returns available model pipelines, thresholds, and feature dimensionalities.
- `POST /predict`: Accepts single transaction JSON payloads, executes schema validation, returns anomaly scores, flags, and contextual indicators, and writes an audit log.
- `POST /predict-batch`: Vectorized batch scoring for transaction arrays, calculating batch latencies and summary statistics.
- `GET /audit-logs`: Retrieves recent anonymized triage events.

### 8.3 Automated Test Suite (`tests/`)
Comprehensive automated testing is implemented using `pytest`, verifying 30 distinct unit and integration assertions:
- `test_validation.py`: Schema validation, missing fields, negative amounts, invalid categories, duplicate IDs, non-binary labels, quality reports.
- `test_feature_engineering.py`: Target leakage guards (asserting `is_fraud` is absent from $X$), continuous cyclical bounds, log1p transformations, prior card velocity temporal integrity.
- `test_preprocessing.py`: Fit-on-train integrity, handling of novel/unseen test category levels without crashing.
- `test_models.py`: Isolation Forest calibration, LOF novelty scoring, Autoencoder reconstruction MSE, SMOTE in-fold execution, minority-under-2 fallback safety.
- `test_scoring.py`: Uniform score direction verification, single record triage formatting, batch DataFrame enrichment.
- `test_metrics.py`: Standard binary evaluation, single-class edge case handling (ensuring zero divisions never crash PR-AUC or ROC-AUC), sparse minority warnings.
- `test_smoke_app.py`: Verification of FastAPI REST endpoints (`/health`, `/models`, `/predict`, `/predict-batch`, `/audit-logs`).

**Test Execution Result:** **30 passed, 0 failed** in 7.30 seconds.

---

## 9. Ethical Governance, Privacy and Regulatory Compliance

### 9.1 Digital Personal Data Protection (DPDP) Act 2023 Compliance
Under India's Digital Personal Data Protection Act, 2023, financial identifiers and transaction records constitute digital personal data requiring rigorous minimization, purpose limitation, and storage security:
1. **Irreversible Token Masking:** Primary Account Numbers (PANs), Cardholder Verification Values (CVVs), and expiration dates are never ingested, requested, or persisted. All simulated accounts use masked tokens (`CARD_XXXX####`).
2. **Audit Minimization:** The SQLite audit ledger (`outputs/audit.db`) stores only mathematical anomaly scores, transaction sequence numbers, model versions, and execution latencies. No personal cardholder attributes or merchant dispute texts are stored in log files.

### 9.2 Ethical Impact of False Positives
In automated financial systems, a False Positive occurs when an anomaly model flags an authorized, legitimate customer purchase. If an algorithm is granted autonomous decline authority:
- Legitimate cardholders suffer embarrassment, denial of essential medical or travel services, and customer attrition.
- Disproportionate impact may fall on demographic groups with non-standard working hours or irregular spending intervals.

To eliminate this ethical risk, our prototype enforces a strict **Human-in-the-Loop Decision Support** doctrine:
> **MANDATORY POLICY:** An anomaly flag is a risk triage signal routing records to a certified fraud investigator. It NEVER triggers an automatic card block, payment rejection, or credit penalty.

---

## 10. Four-Week Project Timeline and Milestones

| Week | Phase | Key Activities Completed | Verification Deliverables |
|---|---|---|---|
| **Week 1** | Problem Formulation & Data Audit | Ingested `dataset.csv`; validated schema; completed `data_dictionary.csv`; generated four reproducible EDA visualizations (`outputs/figures/`). | Problem statement approved; data dictionary completed; EDA script verified. |
| **Week 2** | Architecture & Core Feature Engineering | Implemented `data_validation.py`, `feature_engineering.py`, `preprocessing.py`; prevented target leakage; implemented Isolation Forest and LOF (`novelty=True`). | Working core module demo; feature matrix verified with zero leakage. |
| **Week 3** | Neural Models, Integration & Testing | Implemented Hourglass Autoencoder, supervised baselines, and in-fold dynamic SMOTE; constructed pytest suite (30 passing tests); measured empirical metrics. | End-to-end training pipeline working; evaluation metrics recorded in JSON/CSV. |
| **Week 4** | UI, REST API, Governance & Final Report | Built 7-tab Streamlit interface; implemented FastAPI REST API; created architecture/DFD diagrams; completed 10-page academic report, synopsis, and demo script. | Complete runnable project pack ready for evaluation and expo demo. |

---

## 11. Conclusions and Future Work

### 11.1 Project Conclusions
1. Anomaly detection models provide a practical, unsupervised mechanism for surfacing suspicious transactions in imbalanced financial datasets without relying on extensive labelled historical fraud datasets.
2. Local Outlier Factor (`novelty=True`) achieved the highest detection recall (100.0%) on the held-out evaluation set, successfully capturing the fraud instance at a manageable 7.07% False Positive Rate.
3. Isolation Forest demonstrated exceptional continuous ranking ability ($\text{ROC-AUC} = 0.9899, \text{PR-AUC} = 0.5000$) with minimal false alarms ($\text{FPR} = 1.01\%$).
4. Evaluating models on extreme class imbalance requires Precision-Recall curves and cost-sensitive matrices; accuracy is misleading and must never be reported in isolation.
5. In-fold SMOTE oversampling provides marginal stability improvements for supervised classifiers, but does not alter fundamental sample sparsity when total minority examples are fewer than ten.

### 11.2 Future Enhancements
- **Graph Neural Networks (GNNs):** Constructing bipartite card-merchant transaction graphs to detect coordinated fraud rings and card-testing velocity attacks.
- **Explainable AI (SHAP / TreeSHAP):** Generating local Shapley feature attribution values to explain why a specific transaction exceeded isolation thresholds.
- **Streaming Ingestion:** Adapting the pipeline to Apache Kafka and streaming Faust workers for sub-10 millisecond wire-speed scoring.

---

## 12. References and Academic Bibliography

1. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). *Isolation Forest*. Eighth IEEE International Conference on Data Mining (ICDM), pp. 413–422. DOI: 10.1109/ICDM.2008.17.
2. **Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J.** (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD Record, 29(2), pp. 93–104. DOI: 10.1145/335191.335388.
3. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research (JAIR), 16, pp. 321–357. DOI: 10.1613/jair.953.
4. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830. [https://scikit-learn.org/](https://scikit-learn.org/)
5. **Lemaître, G., Nogueira, F., & Aridas, C. K.** (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning*. Journal of Machine Learning Research (JMLR), 18(17), pp. 1–5.
6. **Government of India.** (2023). *The Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023)*. The Gazette of India, Extraordinary, Part II, Section 1. [https://egazette.gov.in/WriteReadData/2023/249154.pdf](https://egazette.gov.in/WriteReadData/2023/249154.pdf)
7. **Machine Learning Group — Université Libre de Bruxelles (ULB).** (2015). *Credit Card Fraud Detection Benchmark*. Kaggle Dataset Repository. [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
8. **Enterprise Fraud Prevention Standards.** (2025). *Guidelines for Financial Transaction Anomaly Detection and Human Review Workflows*. Financial Data & AI Institute.

---

## Appendix A: AI Tool Use Disclosure
In accordance with engineering integrity and transparency guidelines:
- **AI Assistants Utilized:** Google Gemini and Claude were utilized as coding assistants for project planning, structural brainstorming, test scaffolding, and initial documentation drafting.
- **Learner Responsibility:** All implemented Python modules (`data_validation.py`, `feature_engineering.py`, `preprocessing.py`, `train_models.py`, `scoring.py`, `evaluation.py`, `persistence.py`, `app.py`, `api/main.py`), test assertions, empirical model evaluations, hyperparameter calibrations, and final report conclusions were independently reviewed, executed, verified, and debugged by the learner. No model metrics were fabricated.

---

## Appendix B: Deliverables Inventory
- `app.py` — 7-tab interactive Streamlit web dashboard.
- `api/main.py` — FastAPI REST application with `/health`, `/models`, `/predict`, `/predict-batch`.
- `src/` — 9 modular Python scripts covering validation, feature engineering, models, scoring, evaluation, and persistence.
- `tests/` — 7 pytest test modules containing 30 verified automated test cases.
- `data/dataset.csv` — Immutable original 500-record ground truth dataset.
- `data/dataset_cleaned.csv` — Standardized normalized dataset.
- `data/data_dictionary.csv` — Completed schema reference.
- `data/synthetic_transactions_only.csv` — 4,500 isolated synthetic demonstration records.
- `data/synthetic_data_manifest.json` — Generation manifest.
- `outputs/metrics/model_comparison.csv` — Measured empirical evaluation metrics.
- `outputs/metrics/evaluation_results.json` — Complete JSON evaluation payload.
- `outputs/figures/` — 8 diagnostic figures (EDA distributions, confusion matrices, comparison charts).
- `docs/` — Project report, synopsis, ethical compliance declaration, architecture diagrams, and demo script.
