# 📉 Telco Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco%20Churn-informational)
![Rows](https://img.shields.io/badge/Rows-7%2C032%20(cleaned)-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Key Insights](#key-insights)
7. [Statistical Testing](#statistical-testing)
8. [Predictive Modelling](#predictive-modelling)
9. [Business Recommendations](#business-recommendations)

---

## Project Overview

Customer retention is one of the most critical metrics for any subscription-based business. This project analyses customer behaviour in the telecommunications sector to identify the **primary drivers of churn** (service cancellation), statistically validate those findings, and build a suite of **machine learning models** to flag at-risk customers before they cancel.

The analysis moves through four distinct stages:

1. **Data Cleaning** — handling type coercions and missing values
2. **Exploratory Data Analysis (EDA)** — visualising churn patterns across key features
3. **Statistical Testing** — proving patterns are significant, not coincidental
4. **Predictive Modelling** — comparing four ML models to build an early-warning system

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM Sample Dataset — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Industry | Telecommunications |
| Raw rows | 7,043 |
| Rows after cleaning | 7,032 (11 removed — blank `TotalCharges`) |
| Features | 20 (after dropping `customerID`) |
| Target variable | `Churn` (Yes / No) |
| Baseline churn rate | ~26.5% |

**Feature categories:**

- **Demographics** — `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info** — `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services** — `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Structure

```
telco-churn-analysis/
│
├── telco_customer_churn_analysis.py   # Full analysis script
├── Telco_Customer_Churn_Analysis.ipynb  # Notebook version with narrative
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
└── outputs/                           # Generated charts (after running)
    ├── chart1_churn_contract.png
    ├── chart2_charges_tenure.png
    ├── chart3_tenure_distribution.png
    ├── chart4_internet_techsupport.png
    ├── chart5_internet_techsupport.png    
    ├── chart6_heatmap.png
    ├── chart7_distribution_skewness.png
    ├── chart8_statistical_significance.png
    ├── chart9_model_comparison.png
    ├── chart10_confusion_matrices.png
    └── chart11_feature_importance.png
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/telco-churn-analysis.git
cd telco-churn-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

**As a Python script:**
```bash
python telco_customer_churn_analysis.py
```

**As a Jupyter Notebook:**
```bash
jupyter notebook Telco_Customer_Churn_Analysis.ipynb
```

> **Note:** Make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the same directory as the script, or update `DATA_PATH` at the top of the script to point to its location.

---

## Analysis Pipeline

```
Raw CSV (7,043 rows)
        │
        ▼
  Data Cleaning
  • Drop customerID
  • Coerce TotalCharges → float
  • Drop 11 rows with blank TotalCharges
        │
        ▼
  Exploratory Data Analysis
  • Churn distribution & contract type
  • Monthly charges & tenure boxplots
  • Tenure danger-zone histogram
  • Internet service & tech support breakdown
  • Numerical correlation heatmap
        │
        ▼
  Statistical Testing
  • Chi-Square (6 categorical variables)
  • Skewness check → Mann-Whitney U (3 numerical variables)
  • Significance summary chart
        │
        ▼
  Predictive Modelling
  • 4 models trained on identical stratified splits
  • Compared by Recall, F1, ROC-AUC, Precision, Accuracy
  • Confusion matrices for all models
  • Feature importance (Random Forest)
```

---

## Key Insights

### 📋 The Contract Trap
Customers on **month-to-month contracts** churn at a dramatically higher rate than those on 1-year or 2-year contracts. The absence of a commitment makes switching costs near zero. This is the single strongest predictor of churn in the dataset.

### ⏱ The New Customer Danger Zone
Churn is heavily concentrated in a customer's **first 12 months**. Customers who survive past the one-year mark show dramatically higher loyalty. This makes early onboarding and first-year retention incentives the highest-ROI interventions available.

### 💰 The Premium Service Paradox
**Fiber Optic** customers — the highest-paying segment — churn at a significantly higher rate than DSL customers. This suggests either a perceived value gap (the premium price isn't felt to be justified) or a reliability/quality problem with the Fiber tier specifically.

### 🛠 The Tech Support Lifeline
Customers **without Tech Support** churn at nearly twice the rate of those who have it. Technical frustration is a primary churn driver, making support bundling — especially for new and Fiber Optic customers — a directly actionable retention lever.

---

## Statistical Testing

All EDA findings were validated with formal statistical tests. Test selection was justified by a prior skewness check rather than assumed.

| Variable | Test Used | Result |
|---|---|---|
| Contract | Chi-Square | ✅ Significant |
| InternetService | Chi-Square | ✅ Significant |
| TechSupport | Chi-Square | ✅ Significant |
| PaymentMethod | Chi-Square | ✅ Significant |
| PaperlessBilling | Chi-Square | ✅ Significant |
| OnlineSecurity | Chi-Square | ✅ Significant |
| MonthlyCharges | Mann-Whitney U | ✅ Significant |
| tenure | Mann-Whitney U | ✅ Significant |
| TotalCharges | Mann-Whitney U | ✅ Significant |

**Why Mann-Whitney U for numerical variables?**
A skewness check confirmed that `MonthlyCharges` and `TotalCharges` are non-normally distributed. Mann-Whitney U makes no normality assumption, making it the appropriate test. A Student's t-test would have been statistically invalid here.

**Significance threshold:** p < 0.05. All tested variables cleared this bar comfortably (most with p-values below 1e-10), meaning the findings are robust even under stricter corrections.

---

## Predictive Modelling

Four models were trained on an identical 80/20 stratified train/test split and evaluated on the metrics that matter for an imbalanced churn problem.

> ⚠️ **Why not just use accuracy?** A model that predicts "No Churn" for every customer achieves ~74% accuracy while being completely useless. **Recall** — catching as many real churners as possible — is the primary business metric. A missed churner means lost revenue with no chance to intervene.

### Models Compared

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline. Scaled features via Pipeline. Interpretable coefficients. |
| **Decision Tree** | Depth-capped at 8. Highly interpretable; can be visualised as a flowchart. |
| **Random Forest** | 200 trees, `class_weight="balanced"`. Stable feature importances. |
| **Gradient Boosting** | Conservative LR + subsampling for imbalance. Typically strongest on tabular data. |

All tree-based models use `class_weight="balanced"` or equivalent tuning to correct for the ~74/26 class imbalance. Without this, models silently over-predict "No Churn."

### Class Imbalance Handling

| Approach | Applied To |
|---|---|
| `class_weight="balanced"` | Logistic Regression, Decision Tree, Random Forest |
| Conservative `learning_rate` + `subsample=0.8` | Gradient Boosting |
| `stratify=y` on train/test split | All models |

### Top Feature Importances (Random Forest)

The Random Forest model identifies `tenure`, `MonthlyCharges`, and `Contract_Month-to-month` as the top predictors of churn — consistent with the EDA findings and providing a direct link between the statistical analysis and the model's decision-making.

---

## Business Recommendations

| Priority | Action | Insight It Addresses |
|---|---|---|
| 🔴 High | Offer first-year loyalty discounts to migrate month-to-month customers to annual contracts | Contract Trap |
| 🔴 High | Implement a proactive 90-day onboarding programme for new customers | Danger Zone |
| 🟡 Medium | Bundle free or discounted Tech Support into new customer packages, especially Fiber Optic | Tech Support Lifeline |
| 🟡 Medium | Deploy the churn prediction model to flag at-risk customers for the retention team | Predictive modelling output |
| 🟢 Low | Audit Fiber Optic service quality and pricing — investigate support ticket volume for this segment | Premium Service Paradox |
