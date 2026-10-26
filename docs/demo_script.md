# 1–2 Minute Live Demo & Presentation Walkthrough Script
## FraudGuard AI: Enterprise Credit Card Fraud Anomaly Detection Platform
**Presenter:** Swapnil Bhardwaj | Enterprise AI / AIML Solutions

---

### [0:00 – 0:25] Opening Hook & Problem Statement
> *"Good morning, evaluators and mentors. In modern payment processing, credit card fraud is a severe rare-event problem. In our ground-truth dataset of 500 transactions, only 6 are fraudulent — just 1.2%. If a system naively predicts 'legitimate' for every transaction, it achieves 98.8% accuracy while catching zero fraud.*  
> 
> *Our goal was to build a leakage-free, reproducible anomaly detection prototype that flags unusual transactions for human compliance review without ever automatically declining a cardholder's transaction."*

---

### [0:25 – 0:50] Architecture & Methodological Discipline (Screen 1 & 2)
> *(Navigate to Streamlit Tab 1 & Tab 3)*  
> *"Here in our Streamlit application, our Overview tab audits the schema and data quality. We implemented strict methodological safeguards:*  
> *First: Zero target leakage. The `is_fraud` label is strictly excluded from all feature engineering.*  
> *Second: All scalers and frequency encoders are fitted strictly on the 80% training split.*  
> *Third: We compared five modeling approaches: Isolation Forest, Local Outlier Factor configured for novelty detection on unseen data, a Neural Autoencoder measuring reconstruction error, and a supervised baseline with within-fold SMOTE oversampling."*

---

### [0:50 – 1:20] Empirical Findings & Model Comparison (Screen 3)
> *(Navigate to Streamlit Tab 4)*  
> *"Looking at our held-out test evaluation on 100 unseen transactions:  
> **Local Outlier Factor** was our top-performing detection model. It caught the held-out fraud transaction — achieving **100% Recall** with an **F1-score of 0.2222** and an **ROC-AUC of 0.9596**. It filtered out 92 legitimate transactions automatically, leaving just 8 items for human triage.  
> Meanwhile, **Isolation Forest** achieved an outstanding **ROC-AUC of 0.9899** and **PR-AUC of 0.5000** with only a 1% False Positive Rate.  
> Notice that our test suite of 30 unit tests validates these exact calculations."*

---

### [1:20 – 1:50] Live Scoring & Ethical Governance (Screen 4 & 5)
> *(Navigate to Streamlit Tab 5 & Tab 7)*  
> *"Let's test an individual transaction in real time. We enter an unusual late-night online electronics transaction of \$1,850. In under 6 milliseconds, the model flags it with an anomaly score of 1.15 against a threshold of 1.14, recommending human review and surfacing non-causal contextual risk indicators.  
> 
> In accordance with the Digital Personal Data Protection Act 2023, no raw card numbers, CVVs, or customer names are ever processed or stored. Our SQLite audit trail logs only mathematical anomaly scores for compliance traceability.*  
> 
> *In summary, this prototype delivers a verified, decision-support solution balancing fraud protection with cardholder convenience. Thank you, and I look forward to your questions."*
