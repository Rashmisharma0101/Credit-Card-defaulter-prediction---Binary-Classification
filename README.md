# Credit-Card-defaulter-prediction---Binary-Classification (UCI Dataset)

## Project Overview
This project focuses on predicting credit card default using the **UCI Credit Default dataset**.  
The objective is to build a machine learning pipeline that balances **recall** (catching as many defaulters as possible) and **precision** (minimizing false alarms).  

In real-world applications, this type of modeling is used to **rank customers by default risk**, helping financial institutions decide which accounts require closer monitoring or intervention.

---

## Dataset
- Source: [UCI Machine Learning Repository – Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- Size: 30,000 customers, 23 features  
- Features include:
  - **Demographics**: Age, Gender, Education, Marriage
  - **Credit history**: Credit limit, past payment status (PAY_0 … PAY_6)
  - **Billing & Payments**: Bill amounts and payments for last 6 months
- Target: `default.payment.next.month` (1 = Default, 0 = No Default)

---

## Feature Engineering
To capture repayment behavior, the following features were engineered:
- **Repayment Ratios**: `PAY_AMT / BILL_AMT` for each month
- **Chronic Underpayer**: Consistently paying below a threshold % of bills
- **Zero Pay Months**: Count of months with no payment
- **Payment Drop-off**: Month when payments stopped
- **Dropoff Risk**: Higher risk for earlier payment drop-offs
- **Age Bins**: Grouped customer ages

---

## Modeling
- Baseline models: Logistic Regression
- Hyperparameter tuning with `RandomizedSearchCV`- XGBoost Classifier
- Evaluation metrics:
  - **Recall (class 1)**: Prioritized, since missing defaulters is costly
  - **Precision (class 1)**: Monitored, to reduce false positives
  - **F1-score & PR Curve**: To assess balance between precision and recall
- Threshold tuning and **Top-K alerting strategy** were applied

---

## Results
- At **Recall ≥ 80%**, best achievable Precision ≈ **33%**
- Demonstrates the inherent **precision–recall trade-off** in imbalanced credit risk tasks
- Real-world implication: Best use is as a **risk ranking system**, where banks investigate the **top 5–10% most risky customers**

---

Model can be deployed as a **risk scoring tool** instead of strict classifier
