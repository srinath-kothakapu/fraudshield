# Interview Guide for FraudShield

## 30-second pitch
"I built an end-to-end fraud detection system. Instead of optimising F1, I modelled the *business cost*
of missed fraud vs. false alarms and tuned the decision threshold to minimise it, which cut simulated losses 40%
on a chronologically held-out test set. It's calibrated, explainable with SHAP reason codes, monitored for drift with PSI,
and served through a tested, Dockerised FastAPI with CI."

## Likely questions and strong answers
**Why not accuracy?** 1.5% fraud, so "approve everything" is 98.5% accurate and useless. I use PR-AUC and dollar cost.

**Why a time-based split?** Fraud patterns evolve. Random splits leak future info and overstate performance.

**How did you choose the threshold?** Minimised expected cost: missed fraud = amount + chargeback fee; false alarm = review + friction; every alert has a review cost. Threshold tuned on validation, reported once on test.

**Why calibrate?** Cost-based thresholds only make sense if probabilities mean something. Isotonic regression on a held-out set; checked with a reliability curve.

**Imbalance handling?** `scale_pos_weight`/class weights plus the cost-based threshold, rather than SMOTE (which distorts probabilities).

**Logistic regression saved more than XGBoost on validation. Why pick XGBoost?** (Be upfront.) I selected on PR-AUC, which measures ranking quality across all thresholds; LR was better at one high-recall region. In production I'd compare at the operating point, factor in latency/interpretability, and A/B test.

**How do you handle drift?** PSI (<0.1 ok, 0.1–0.25 watch, >0.25 retrain) + KS on every feature vs. the training reference; next step is performance monitoring once labels mature.

**Why synthetic data?** Reproducible, no PII, lets me inject known patterns and controlled drift to *prove* the monitor works. The pipeline is data-agnostic; I can point it at a real dataset.

**What would you do with more time?** MLflow tracking, Optuna, delayed-label handling, shadow deployment, latency/load tests, fairness checks.

## Things to demo live
1. `make all`, 2. open `/docs`, send a risky vs. safe transaction, 3. show SHAP reasons, 4. show `reports/drift_drifted.csv`.
