# Ethical Compliance and Data Governance Declaration
## FraudGuard AI Platform | Digital-Data-AI / AIML
**Project Title:** FraudGuard AI — Enterprise Credit Card Fraud Detection Using Anomaly Detection  
**Lead Developer:** Swapnil Bhardwaj  
**Roll Number:** [ENTER ROLL NUMBER]  
**College / Campus:** ABES Engineering College  
**Academic Year:** 2025–26  

---

### Statement of Ethical Compliance

As a developer and data scientist building the FraudGuard AI platform, I declare that:

1. **Dataset Integrity and Provenance:**
   - The primary input file `dataset.csv` has been retained in its original ground-truth form. No rows were altered, fabricated, or deleted.
   - Any supplementary records generated for demonstration purposes have been placed in a dedicated, clearly separated file (`data/synthetic_transactions_only.csv`) accompanied by `synthetic_data_manifest.json`, certifying that they are artificial.
   - The dataset schema is explicitly distinguished from the public Kaggle ULB benchmark, and no false provenance claims have been made.

2. **Protection of Personally Identifiable Information (PII):**
   - The software does not request, collect, process, or store unmasked Primary Account Numbers (PANs), Cardholder Verification Values (CVVs), expiration dates, PINs, cardholder full names, billing addresses, or phone numbers.
   - All card references use surrogate masked tokens (`CARD_XXXX####`).
   - The application audit database (`outputs/audit.db`) stores only anonymized execution telemetry (transaction ID, model name, anomaly score, decision threshold, timestamp, and latency).

3. **Human-in-the-Loop Decision Support (No Automated Declines):**
   - Anomaly detection outputs are designed strictly as decision-support risk signals to assist human compliance analysts in prioritizing reviews.
   - The application does not automatically decline transactions, freeze accounts, or execute unilateral financial decisions that could disrupt legitimate consumer commerce or restrict essential financial access.

4. **Compliance with the Digital Personal Data Protection (DPDP) Act 2023:**
   - Data handling conforms to the principles of purpose limitation, storage limitation, and data minimization as articulated in the DPDP Act 2023.

5. **Transparency in AI Tool Usage:**
   - The use of AI coding assistants (Google Gemini and Claude) for code scaffolding, test generation, and documentation drafting has been fully disclosed. All final code, models, and empirical results have been verified, executed, and tested by the learner.

---

### Verification Checklist

- [x] Ground-truth `dataset.csv` preserved unchanged.
- [x] Masked tokens used exclusively; zero CVV, PAN, or PII storage.
- [x] Automated decline capability disabled; human review workflow enforced.
- [x] Synthetic records isolated in separate file with manifest.
- [x] AI assistance transparently disclosed.

**Learner Signature:** ______________________________________  
**Date:** ________________________  
**Mentor Signature:** ______________________________________  
**Date:** ________________________  
