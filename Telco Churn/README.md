# 📉 Telco Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-enabled-blueviolet)
![lifelines](https://img.shields.io/badge/lifelines-survival%20analysis-teal)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco%20Churn-informational)
![Rows](https://img.shields.io/badge/Rows-7%2C032%20cleaned-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Section 1 — Data Cleaning](#section-1--data-cleaning)
7. [Section 2 — Exploratory Data Analysis](#section-2--exploratory-data-analysis)
8. [Section 3 — Statistical Testing](#section-3--statistical-testing)
9. [Section 4 — Predictive Modelling](#section-4--predictive-modelling)
10. [Section 5 — Advanced Analytics](#section-5--advanced-analytics)
11. [Key Findings](#key-findings)
12. [Business Recommendations](#business-recommendations)
13. [Dependencies](#dependencies)

---

## Project Overview

Customer retention is one of the most critical metrics for any subscription-based business. This project analyses customer behaviour in the telecommunications sector to:

- Identify the **primary drivers of churn** through exploratory analysis and statistical validation
- Build and compare **four machine learning models** to predict which customers are likely to cancel
- Apply **advanced analytical techniques** — cohort analysis, retention curves, survival analysis, CLV risk segmentation, and SHAP-based churn driver profiling — to move beyond descriptive statistics into operationally actionable intelligence

The end goal is a prioritised **retention queue**: a ranked list of at-risk customers with revenue figures attached, enabling the business to deploy retention spend where it has the highest return.

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM Sample Dataset — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Industry | Telecommunications |
| Raw rows | 7,043 |
| Rows after cleaning | 7,032 |
| Rows removed | 11 (blank `TotalCharges` — new customers with zero tenure) |
| Features | 20 (after dropping `customerID`) |
| Target | `Churn` — Yes / No |
| Baseline churn rate | ~26.5% |

**Feature categories:**

- **Demographics** — `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account** — `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services** — `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## Project Structure

```
telco-churn-analysis/
│
├── Telco.ipynb                              # Main analysis notebook (67 cells)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Raw dataset
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
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

> `shap` and `lifelines` are installed automatically by the notebook's first cell if not already present.

### 3. Update the data path
In Cell 1 of the notebook, update `DATA_PATH` to point to the CSV on your machine:
```python
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

### 4. Run the notebook
```bash
jupyter notebook Telco.ipynb
```
Run all cells top to bottom. Sections 1–4 are fully independent. Section 5 requires the objects produced by Section 4 to be in memory.

---

## Analysis Pipeline

```
Raw CSV (7,043 rows)
        │
        ▼
 1. Data Cleaning          Drop ID, coerce TotalCharges, remove 11 nulls
        │
        ▼
 2. EDA                    11 chart sets across contract, tenure, services,
                           payment method, charges, and billing
        │
        ▼
 3. Statistical Testing    Chi-Square (6 categorical) + skewness check +
                           Mann-Whitney U (3 numerical) — all 9 significant
        │
        ▼
 4. Predictive Modelling   4 models, stratified split, recall-first evaluation,
                           confusion matrices, feature importance
        │
        ▼
 5. Advanced Analytics     Cohort analysis → Retention curves →
                           CLV risk segmentation → Survival analysis →
                           Churn risk scoring & retention queue
```

---

## Section 1 — Data Cleaning

- `customerID` dropped — no analytical or predictive value
- `TotalCharges` coerced from string to float — 11 rows with blank values removed
- Final shape: **7,032 rows × 20 features**
- Baseline churn rate: **26.5%** (1,869 churned / 7,032 total)

---

## Section 2 — Exploratory Data Analysis

| Chart | What it shows |
|---|---|
| Overall churn distribution | 26.5% baseline with count + % labels |
| Churn by contract type | Month-to-month 42.7% vs 2.8% two-year |
| Monthly charges vs churn | Boxplot + jitter; median $79 churned vs $56 retained |
| Tenure vs churn | Shorter median tenure for churners |
| Tenure histogram | Danger zone 0–12 months annotated |
| Churn by internet service | Fiber Optic 41.9% vs DSL 18.9% |
| Churn by tech support | 41% without vs 15% with support |
| Churn by payment method | Electronic check highest among payment types |
| Churn by paperless billing | Elevated churn among paperless billing users |
| Correlation heatmap | Tenure–TotalCharges collinearity flagged (r ≈ 0.83) |

All grouped bar charts include within-group churn rate % labels — enabling fair comparison across groups of different sizes, not just raw counts.

---

## Section 3 — Statistical Testing

### Objective
Mathematically validate all EDA patterns so findings can be presented to stakeholders as statistically proven rather than visually suggested.

### Test selection justification
A **skewness check** is run before numerical tests. `MonthlyCharges` and `TotalCharges` are confirmed non-normal — Mann-Whitney U is applied rather than a t-test, which would be invalid for skewed distributions.

### Chi-Square Tests — Categorical Variables

| Variable | Result |
|---|---|
| Contract | ✅ Significant |
| InternetService | ✅ Significant |
| TechSupport | ✅ Significant |
| PaymentMethod | ✅ Significant |
| PaperlessBilling | ✅ Significant |
| OnlineSecurity | ✅ Significant |

### Mann-Whitney U Tests — Numerical Variables

| Variable | Result |
|---|---|
| MonthlyCharges | ✅ Significant |
| tenure | ✅ Significant |
| TotalCharges | ✅ Significant |

All nine variables clear p < 0.05 by a wide margin. A significance summary chart visualises all results ranked by −log₁₀(p-value).

---

## Section 4 — Predictive Modelling

### Why Recall, not Accuracy?
A model predicting "No Churn" for every customer achieves ~74% accuracy while being completely useless. **Recall for the Churn class** is the primary metric — a missed churner means lost revenue with no chance to intervene.

### Models & Imbalance Handling

| Model | Class imbalance approach |
|---|---|
| Logistic Regression | `class_weight="balanced"` + `StandardScaler` pipeline |
| Decision Tree | `class_weight="balanced"`, `max_depth=8` |
| Random Forest | `class_weight="balanced"`, 200 trees, all CPU cores |
| Gradient Boosting | `learning_rate=0.05`, `subsample=0.8` |

All models trained on an identical 80/20 stratified split.

### Outputs
- Model comparison across Recall, Precision, F1, ROC-AUC, Accuracy
- Confusion matrices for all four models side by side
- Top 15 feature importances (Random Forest)
- Full classification report for the best-performing model

---

## Section 5 — Advanced Analytics

### A — Cohort Analysis
Customers grouped into tenure bands (0–12, 13–24, 25–48, 49–72 months) as a proxy for joining period. Three-panel chart shows churn rate per cohort, average monthly charges, and contract mix. A cohort × contract heatmap identifies the single highest-risk segment combination.

**Key finding:** Newer cohorts churn at a far higher rate, driven by their concentration on month-to-month contracts — not by inherent customer quality differences.

### B — Retention Curves
Month-by-month retention curves for each contract type and internet service group. Correctly treats non-churned customers as still-active (not excluded). The 0–12 month danger zone is annotated. Final retention percentages labelled at the curve endpoint.

### C — Customer Lifetime Value Risk Segmentation
Two CLV components per customer:
- **Realised CLV** = `TotalCharges` (revenue already captured)
- **Projected CLV** = `MonthlyCharges × estimated remaining tenure`

Remaining tenure estimated from the median tenure of retained customers within each contract group — conservative and anchored to observed behaviour.

A **2×2 matrix** (CLV tier × churn probability tier) produces four segments:

| | Low Risk | High Risk |
|---|---|---|
| **High CLV** | STABLE | **CRITICAL 🚨** |
| **Low CLV** | HEALTHY | MONITOR |

The **CRITICAL** segment is the primary output — high-value customers the model flags as likely to leave, with total revenue at risk computed per segment.

### D — Survival Analysis (Kaplan-Meier)
KM correctly handles right-censoring: customers who haven't churned yet contribute information about minimum survival time rather than being excluded. Survival curves stratified by contract type and internet service with 95% confidence intervals. **Log-rank test p-values** annotated directly on each chart confirming curve differences are statistically significant.

### E — Churn Risk Scoring & Segmentation
Every customer scored (0–1) by the best-performing model and bucketed into four tiers:

| Tier | Threshold | Action |
|---|---|---|
| CRITICAL | ≥ 0.75 | Immediate personal outreach — senior retention team |
| HIGH | ≥ 0.50 | Automated targeted offer within 48 hours |
| MEDIUM | ≥ 0.25 | Soft-touch nurture campaign |
| LOW | < 0.25 | No active intervention required |

A **tier validation chart** confirms calibration: actual churn rates increase monotonically LOW → CRITICAL, proving the scores are trustworthy for prioritisation. A normalised profile heatmap characterises each tier across five dimensions for stakeholder briefing.

---

## Key Findings

**The Contract Trap** — Month-to-month customers churn at 42.7% — nearly 15× the rate of two-year contract holders. Contract type is the single strongest predictor of churn in the dataset.

**The New Customer Danger Zone** — Over 55% of all churners leave within their first 12 months. Customers who survive past year one show dramatically higher loyalty. Early retention investment has the highest ROI of any intervention.

**The Premium Service Paradox** — Fiber Optic customers churn at 41.9% despite paying the highest monthly charges. This signals either a service quality problem or a value-for-price gap at the premium tier that competitors are exploiting.

**The Tech Support Lifeline** — Customers without Tech Support churn at 41% vs 15% for supported customers — a 2.7× difference that is statistically confirmed. Bundling support is a directly actionable, high-return retention strategy.

**The Electronic Check Signal** — Electronic check users show the highest churn rate among all payment methods, suggesting a correlation between payment friction and broader customer disengagement.

**The CRITICAL Segment** — The CLV × risk model identifies a specific, named group of high-value customers with elevated churn probability. These customers represent a disproportionate share of projected revenue at risk and are the primary target for retention investment.

---

## Business Recommendations

| Priority | Action | Driven by |
|---|---|---|
| 🔴 High | Deploy churn model — flag CRITICAL customers weekly for retention team | Risk segmentation |
| 🔴 High | Offer first-year loyalty discounts to migrate month-to-month → annual contracts | Contract trap |
| 🔴 High | Implement proactive 90-day onboarding programme for all new customers | Danger zone |
| 🟡 Medium | Bundle Tech Support into Fiber Optic entry packages at no added cost | Tech support lifeline |
| 🟡 Medium | Audit Fiber Optic service quality — investigate NPS and support ticket volume | Premium paradox |
| 🟡 Medium | Incentivise electronic check users to switch to automatic payment | Electronic check signal |
| 🟢 Low | Use SHAP profiles to personalise retention offers per customer risk driver | SHAP analysis |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
shap>=0.43.0
lifelines>=0.27.0
jupyter>=1.0.0
```

```bash
pip install -r requirements.txt
```

`shap` and `lifelines` are also auto-installed by the notebook's first cell if missing.
