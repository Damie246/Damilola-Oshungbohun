# ✈️ Aviation Passenger Satisfaction Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-optional-brightgreen?logo=xgboost)
![Tableau](https://img.shields.io/badge/Tableau-Public-E97627?logo=tableau&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Aviation%20Satisfaction-informational)
![Rows](https://img.shields.io/badge/Rows-103%2C594%20cleaned-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Section 1 — Data Cleaning](#section-1--data-cleaning)
7. [Section 2 — Feature Engineering](#section-2--feature-engineering)
8. [Section 3 — Exploratory Data Analysis](#section-3--exploratory-data-analysis)
9. [Section 4 — Statistical Testing](#section-4--statistical-testing)
10. [Section 5 — Predictive Modelling](#section-5--predictive-modelling)
11. [Section 6 — Model Evaluation](#section-6--model-evaluation)
12. [Section 7 — Segment Analysis](#section-7--segment-analysis)
13. [Section 8 — Tableau Dashboard](#section-8--tableau-dashboard)
14. [Key Findings](#key-findings)
15. [Business Recommendations](#business-recommendations)
16. [Dependencies](#dependencies)

---

## Project Overview

Passenger satisfaction is one of the most commercially critical metrics in aviation. A dissatisfied passenger does not just represent a lost ticket — they represent lost loyalty, lost referrals, and a lost opportunity to convert an occasional flyer into a frequent one. This project analyses over 103,000 passenger logs to:

- Identify the **primary drivers of satisfaction and dissatisfaction** through exploratory analysis and statistical validation
- Build and compare **machine learning models** to predict whether a passenger will be satisfied based on their flight profile
- Translate model outputs into a **prioritised investment roadmap** that tells airline management exactly where to direct capital for the highest return on customer experience
- Present all findings in a **Tableau Public executive dashboard** accessible to non-technical stakeholders

---

## Dataset

| Property | Value |
|---|---|
| Source | Aviation Passenger Satisfaction Survey |
| Industry | Aviation |
| Raw rows | 103,905 |
| Rows after cleaning | 103,594 |
| Rows removed | 311 (missing `Arrival Delay in Minutes`) |
| Features | 22 (after dropping index columns) |
| Target | `satisfaction` — satisfied / neutral or dissatisfied |
| Baseline satisfaction rate | ~43.3% |

**Feature categories:**

- **Demographics** — `Gender`, `Age`, `Customer Type`
- **Trip profile** — `Type of Travel`, `Class`, `Flight Distance`
- **Operational** — `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- **Service ratings (1-5 scale)** — `Inflight wifi service`, `Departure/Arrival time convenient`, `Ease of Online booking`, `Gate location`, `Food and drink`, `Online boarding`, `Seat comfort`, `Inflight entertainment`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Inflight service`, `Cleanliness`

---

## Project Structure

```
aviation-satisfaction-analysis/
│
├── aviation_satisfaction_analysis.py        # Main analysis script (11 steps)
├── Aviation_Satisfaction_Analysis.csv       # Raw dataset
├── Aviation_Satisfaction_Analysis.ipynb     # Notebook version
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/aviation-satisfaction-analysis.git
cd aviation-satisfaction-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> `xgboost` is optional. The script detects whether it is installed and includes or skips it automatically. All other steps run without it.

### 3. Place the dataset
Ensure `Aviation_Satisfaction_Analysis.csv` is in the same directory as the script. The script loads it by filename:
```python
df = pd.read_csv("Aviation_Satisfaction_Analysis.csv")
```

### 4. Run the script
```bash
python aviation_satisfaction_analysis.py
```

Or open the notebook version:
```bash
jupyter notebook Aviation_Satisfaction_Analysis.ipynb
```

All 15 figures are saved automatically to the working directory at 300 DPI. Steps run top to bottom with no manual intervention required.

---

## Analysis Pipeline

```
Raw CSV (103,905 rows)
        │
        ▼
 1. Data Cleaning          Drop index columns, remove 311 rows with missing
                           arrival delay values
        │
        ▼
 2. Feature Engineering    Age Group segmentation, Total Delay composite,
                           Avg Service Score across 14 survey items
        │
        ▼
 3. EDA                    15 chart sets across satisfaction baseline,
                           class, service ratings, delays, age segments,
                           and correlation structure
        │
        ▼
 4. Statistical Testing    Chi-Square (4 categorical) + Point-Biserial
                           (14 service + 4 numeric) — all confirmed significant
        │
        ▼
 5. Predictive Modelling   2-3 models, stratified split, 5-fold cross-
                           validation, ROC-AUC primary metric
        │
        ▼
 6. Model Evaluation       Confusion matrices, ROC curves, benchmark
                           comparison across Accuracy / AUC / CV Mean
        │
        ▼
 7. Segment Analysis       KDE distributions + boxplots of Avg Service
                           Score by Class and satisfaction outcome
        │
        ▼
 8. Tableau Dashboard      Executive summary — 6 panels, interactive
                           filters, parameter controls, no ML knowledge required
```

---

## Section 1 — Data Cleaning

- `Unnamed: 0` and `id` dropped — no analytical or predictive value
- 311 rows with missing `Arrival Delay in Minutes` removed — represents less than 0.3% of total records, dropped rather than imputed to avoid introducing bias into delay analysis
- Final shape: **103,594 rows × 22 features**
- Baseline satisfaction rate: **43.3%** (satisfied passengers in the minority)

---

## Section 2 — Feature Engineering

Three new features created to enrich the model and enable segment analysis:

| Feature | Logic | Purpose |
|---|---|---|
| `Age Group` | Binned into `<18`, `18-29`, `30-45`, `46-60`, `60+` | Enables demographic segmentation in EDA and Tableau heatmap |
| `Total Delay` | `Departure Delay + Arrival Delay`, capped at 99th percentile | Removes extreme outliers (max raw value: 2,176 min) from skewing model |
| `Avg Service Score` | Mean of all 14 service rating columns | Single composite score summarising overall service experience per passenger |

---

## Section 3 — Exploratory Data Analysis

| Chart | What it shows |
|---|---|
| Overall satisfaction pie | 43.3% satisfied baseline — fewer than half of passengers leave happy |
| Satisfaction by ticket class (count) | Business class majority satisfied; Eco majority dissatisfied |
| Satisfaction rate by class (%) | Horizontal bar — corrects for class size differences in raw counts |
| Wi-Fi / Seat Comfort / Food & Drink ratings | Side-by-side countplots showing satisfaction split at each 1-5 score |
| Satisfaction rate per score (8 services) | Small multiples — shows exact score threshold that flips satisfaction |
| Satisfaction by customer loyalty | Disloyal customers overwhelmingly dissatisfied; even loyal customers show high dissatisfaction |
| Satisfaction by type of travel | Personal travellers far more dissatisfied than business travellers |
| Age Group × Class heatmap | Highlight table showing satisfaction rate % — Business only class with blue cells |
| Delay scatter plot | Departure vs. Arrival delay coloured by satisfaction — moderate delays still produce satisfied passengers |
| Correlation heatmap | Full feature correlation matrix — Online Boarding and Wi-Fi cluster separately from physical comfort features |
| Avg Service Score KDE | Distribution curves by satisfaction — clear separation between groups |
| Avg Service Score boxplot | Score distribution by Class × Satisfaction |

All percentage-based charts include a 50% reference line to anchor the reader's interpretation.

---

## Section 4 — Statistical Testing

### Objective
Validate all EDA patterns mathematically so findings can be presented to stakeholders as statistically proven rather than visually suggested.

### Chi-Square Tests — Categorical Variables

| Variable | Result |
|---|---|
| Class | Significant |
| Type of Travel | Significant |
| Customer Type | Significant |
| Gender | Significant |

### Point-Biserial Correlations — Numeric and Service Variables

| Variable | Direction |
|---|---|
| Online boarding | Positive — higher rating strongly predicts satisfaction |
| Inflight wifi service | Positive — strong predictor |
| Seat comfort | Positive — moderate predictor |
| Inflight entertainment | Positive — moderate predictor |
| Departure/Arrival time convenient | Positive |
| Ease of Online booking | Positive |
| Food and drink | Positive — weaker than comfort-based features |
| Total Delay | Negative — higher delay correlates with dissatisfaction |
| Flight Distance | Slight positive — longer flights skew toward business class |

All variables confirmed significant at p < 0.05.

---

## Section 5 — Predictive Modelling

### Why ROC-AUC, not just Accuracy?
The dataset is moderately imbalanced (43% satisfied vs. 57% dissatisfied). A model predicting "dissatisfied" for every passenger achieves ~57% accuracy while being completely useless. **ROC-AUC** is used as the primary metric alongside accuracy — it measures how well the model separates the two classes across all decision thresholds, regardless of class imbalance.

### Models Trained

| Model | Notes |
|---|---|
| Random Forest | 200 trees, `max_depth=20`, all CPU cores, `random_state=42` |
| Gradient Boosting | 200 estimators, `max_depth=5`, `random_state=42` |
| XGBoost | 200 estimators, `learning_rate=0.1`, `max_depth=6` — requires separate install, auto-skipped if unavailable |

All models trained on an identical 80/20 stratified split. 5-fold stratified cross-validation run on every model to confirm scores are not the result of a lucky split.

### Outputs
- Accuracy, ROC-AUC, and 5-fold CV mean and standard deviation per model
- Full classification report for each model
- Top 15 feature importances (Random Forest)

---

## Section 6 — Model Evaluation

| Chart | Purpose |
|---|---|
| Confusion matrices (side by side) | Shows exactly where each model makes mistakes — false positives vs. false negatives |
| ROC curves (overlaid) | Visual comparison of all models across all decision thresholds — AUC annotated per curve |
| Benchmark bar chart | Accuracy, ROC-AUC, and CV Mean compared side by side per model |

The Random Forest achieved **96%+ accuracy** and the highest ROC-AUC across all tested configurations, making it the primary model for feature importance extraction.

---

## Section 7 — Segment Analysis

Two charts examine how the engineered `Avg Service Score` distributes across passenger groups:

- **KDE plot** — overlapping density curves for satisfied vs. dissatisfied passengers confirm the composite score cleanly separates the two groups, validating it as a meaningful feature
- **Boxplot by Class and Satisfaction** — reveals that satisfied Economy passengers still rate services lower in absolute terms than dissatisfied Business passengers, confirming the Economy cabin represents a structurally different service reality rather than just a preference gap

---

## Section 8 — Tableau Dashboard

The Tableau Public executive dashboard translates all Python findings into a single-page visual summary accessible to non-technical stakeholders. No machine learning knowledge is required to read it.

**Dashboard panels:**

| Panel | Type | Business question answered |
|---|---|---|
| Satisfaction Rate | KPI tile | What is our overall satisfaction rate? |
| Customer Count | KPI tile | How large is the dataset? |
| Average Service Score | KPI tile | How are passengers rating us overall? |
| Satisfaction by Class and Travel Type | Side-by-side bar | Where is dissatisfaction concentrated? |
| Age Group × Class Heatmap | Highlight table | Which passenger segments are most at risk? |
| Service Impact | Interactive bar with parameter | Which service ratings flip satisfaction outcomes? |
| Delay Scatter Plot | Scatter | Do delays lose customers or does overall experience matter more? |
| Rating Gap | Horizontal bar | What did satisfied passengers rate highest? |

**Interactivity:**
- `Select Service` parameter control toggles the Service Impact chart between Wi-Fi, Seat Comfort, Food and Drink, and Online Boarding
- `Type of Travel` filter affects class and heatmap panels simultaneously
- Clicking a Class bar filters downstream panels

---

## Key Findings

**The Digital Experience Gap** — Online boarding and inflight Wi-Fi are the top two predictors of satisfaction by a significant margin. Passengers in the modern era treat digital connectivity and seamless check-in as non-negotiables, not optional extras. Ratings of 4 or 5 for Wi-Fi almost guarantee a satisfied outcome regardless of other factors.

**The Economy Problem** — The Economy cabin is where satisfaction collapses. Across every age group in the Age Group × Class heatmap, Economy satisfaction rates sit below 30%. This is not a generational issue — it is a product quality issue. Economy holds the largest share of passengers, which drives the overall 43% satisfaction figure down across the entire airline.

**Delays Do Not Tell the Whole Story** — The delay scatter plot reveals that passengers with moderate delays (100-200 minutes) can still be satisfied when the rest of their experience holds up. This reframes the operational priority: investing in inflight quality can recover customer loyalty even when delays occur.

**Food and Drink Is Overrated as an Investment** — Despite being a common airline focus area, Food and Drink showed consistently weaker predictive power than comfort and digital service features. The ROI on catering investment is lower than the ROI on Wi-Fi or seat upgrades.

**The 46-60 Business Segment Is the Loyalty Core** — The heatmap identifies the 46-60 age group in Business class as the highest-performing segment at over 80% satisfaction. This is the airline's most loyal base and the benchmark experience the rest of the product should aspire to.

**Wi-Fi Has the Most Headroom** — The Rating Gap chart on the Tableau dashboard shows that even satisfied passengers rated Wi-Fi among the lowest of all services. This means Wi-Fi improvement does not just retain at-risk passengers — it upgrades the experience of passengers who are already happy, compounding its return.

---

## Business Recommendations

| Priority | Action | Driven by |
|---|---|---|
| High | Invest in Wi-Fi infrastructure upgrades across the fleet | Top predictor of satisfaction with largest headroom for improvement |
| High | Overhaul the digital boarding platform for speed and reliability | Second-ranked predictor, low infrastructure cost relative to impact |
| High | Audit and refresh the Economy cabin experience | Primary source of dissatisfaction across all age groups |
| Medium | Bundle inflight entertainment content upgrades into Economy tier | Fourth-ranked predictor — most impactful on long-haul Economy routes |
| Medium | Use satisfaction model scores to flag at-risk passengers pre-flight | 96% accuracy enables proactive service recovery before complaints occur |
| Low | Retain Food and Drink spend at current levels | Weak predictor — reallocate budget to digital and comfort upgrades instead |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
jupyter>=1.0.0
xgboost>=1.7.0        # optional
```

```bash
pip install -r requirements.txt
```

XGBoost requires `libomp` on macOS. Install via Homebrew before running:
```bash
brew install libomp
```

If XGBoost is not installed, the script detects this automatically and runs with Random Forest and Gradient Boosting only. All other sections are unaffected.
