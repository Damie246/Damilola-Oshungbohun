# 🌾 Agricultural Crop Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Crop%20Recommendation-informational)
![Rows](https://img.shields.io/badge/Rows-2%2C200%20balanced-lightgrey)
![Crops](https://img.shields.io/badge/Crops-22%20types-yellowgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Step 1 — Data Ingestion and Health Check](#step-1--data-ingestion-and-health-check)
7. [Step 2 — Exploratory Data Analysis](#step-2--exploratory-data-analysis)
8. [Step 3 — Feature Engineering](#step-3--feature-engineering)
9. [Step 4 — Model Training and Evaluation](#step-4--model-training-and-evaluation)
10. [Step 5 — Model Diagnostics and Visualisations](#step-5--model-diagnostics-and-visualisations)
11. [Step 6 — Live Prediction Function](#step-6--live-prediction-function)
12. [Key Findings](#key-findings)
13. [Business Recommendations](#business-recommendations)
14. [Dependencies](#dependencies)

---

## Project Overview

Precision agriculture depends on data-driven decision-making to maximise yield and optimise resource usage. Manual crop selection based on intuition or regional habit frequently results in suboptimal yields, wasted fertiliser, and avoidable crop failure. This project builds an end-to-end machine learning pipeline that:

- Ingests **IoT soil and climate sensor data** and validates its integrity before any modelling begins
- Conducts **exploratory analysis** to map each crop's unique environmental signature across 7 sensor dimensions
- Engineers **4 composite agronomic features** and rigorously tests whether they improve on raw sensor data alone
- Trains and compares **3 models** — a Random Forest baseline, a Random Forest with engineered features, and a Gradient Boosting classifier — using identical splits and 10-fold cross-validation so the comparison is fair and provable
- Delivers a **live prediction function** that accepts real sensor readings and returns a ranked crop recommendation with confidence scores, ready for integration into farm dashboards or IoT pipelines

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Domain | Precision Agriculture / Agronomy |
| Total rows | 2,200 |
| Missing values | None |
| Duplicate rows | None |
| Features | 7 numerical sensor readings |
| Target | `label` — crop type (22 classes) |
| Class balance | Perfectly balanced — exactly 100 readings per crop |

**Sensor features:**

| Feature | Description | Unit |
|---|---|---|
| `N` | Nitrogen content in soil | mg/kg |
| `P` | Phosphorus content in soil | mg/kg |
| `K` | Potassium content in soil | mg/kg |
| `temperature` | Average ambient temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH value | 0–14 scale |
| `rainfall` | Annual rainfall | mm |

**Target classes (22 crops):**
rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

---

## Project Structure

```
crop-recommendation-system/
│
├── crop_recommendation_system.py    # Main analysis and modelling script (7 steps)
├── Crop_recommendation.csv          # Raw dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/crop-recommendation-system.git
cd crop-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Confirm the dataset is in the same directory
The script loads the file by name:
```python
df = pd.read_csv("Crop_recommendation.csv")
```
Ensure `Crop_recommendation.csv` is in the same folder as the script before running.

### 4. Run the script
```bash
python crop_recommendation_system.py
```
The script runs top to bottom. Steps 1–3 are fully independent. Steps 4–6 depend on the objects produced in Step 3 being in memory. All 9 charts are saved automatically as `.png` files in the working directory.

---

## Analysis Pipeline

```
Raw CSV (2,200 rows, 7 features, 22 crops)
        │
        ▼
 1. Data Ingestion & Health Check    Shape, dtypes, nulls, duplicates,
                                     descriptive stats, class distribution
        │
        ▼
 2. Exploratory Data Analysis        5 chart sets: crop distribution,
                                     feature histograms, correlation heatmap,
                                     Nitrogen boxplot by crop, pairplot
        │
        ▼
 3. Feature Engineering              4 composite features: NPK_total,
                                     N_P_ratio, N_K_ratio, climate_index
        │
        ▼
 4. Model Training & Evaluation      3 models on identical stratified splits,
                                     10-fold CV, accuracy delta measured,
                                     per-class classification report
        │
        ▼
 5. Model Diagnostics                Confusion matrix, feature importance,
                                     CV fold chart (all 3 models),
                                     per-class precision bar chart
        │
        ▼
 6. Live Prediction Function         recommend_crop() — accepts 7 sensor
                                     readings, returns top crop + confidence
                                     + top 3 ranked alternatives
```

---

## Step 1 — Data Ingestion and Health Check

- Dataset loaded and audited for shape, column types, missing values, and duplicate rows
- No cleaning required — 0 missing values, 0 duplicate rows
- Descriptive statistics computed for all 7 numerical features
- Target variable confirmed perfectly balanced: **exactly 100 readings per crop across all 22 classes**, eliminating any class-imbalance risk before training

---

## Step 2 — Exploratory Data Analysis

| Chart | What it shows |
|---|---|
| Crop distribution (horizontal bar) | 100 readings per crop confirmed; count labels annotated on each bar |
| Feature distributions (histogram grid) | Shape, spread, and skew of all 7 sensors in a single 2×4 panel |
| Correlation heatmap (lower triangle) | P and K most strongly correlated (0.74); no multicollinearity risk across other pairs |
| Nitrogen by crop (boxplot) | Cotton and Coffee highest; legumes near zero — confirms crops have distinct chemical signatures |
| Pairplot (stratified sample) | 5-feature scatter grid colour-coded by crop; confirms class separability is visually strong |

The correlation heatmap uses a triangular mask to remove redundant mirrored cells. The Nitrogen boxplot is sorted by median to make the contrast between crop groups immediately readable. The pairplot uses a stratified 15-row sample per crop to keep the chart legible at 22 classes.

---

## Step 3 — Feature Engineering

Four composite features were derived from agronomic domain knowledge and added to the original 7 sensors:

| Feature | Formula | Agronomic rationale |
|---|---|---|
| `NPK_total` | `N + P + K` | Total macro-nutrient load — a single proxy for overall soil fertility |
| `N_P_ratio` | `N / (P + ε)` | Nitrogen-to-Phosphorus balance — used in agronomy to diagnose soil composition |
| `N_K_ratio` | `N / (K + ε)` | Nitrogen-to-Potassium balance — signals relative nutrient availability |
| `climate_index` | `temperature × humidity / 100` | Combined heat-moisture stress index |

A small epsilon (1e-6) is added to denominators to prevent division by zero on edge-case readings. These features expand the input space from **7 to 11 features** for the engineered model.

---

## Step 4 — Model Training and Evaluation

### Why Accuracy, not Recall?
Unlike a churn or fraud problem, there is no asymmetric cost between error types here. A false positive (wrong crop recommendation) and a false negative (missing the right one) are equally undesirable for a farmer. Overall accuracy is the appropriate primary metric for a balanced multi-class recommendation task.

### Controlled Baseline Comparison
Three models are trained on **identical 80/20 stratified splits** so test rows are the same across all comparisons:

| Model | Feature set | Test accuracy | CV mean | CV std |
|---|---|---|---|---|
| Random Forest | 7 raw sensors (baseline) | 99.55% | 99.59% | 0.38% |
| Random Forest | 11 engineered features | 99.32% | 99.45% | 0.45% |
| Gradient Boosting | 11 engineered features | 98.64% | 98.82% | 0.79% |

The baseline Random Forest is trained and evaluated fully — not just created. This is the control that makes the comparison meaningful. Without it, any claim that feature engineering "helped" would be unverifiable.

### Key finding
Feature engineering did not improve performance on this dataset. The 7 raw sensor readings already contain enough distinct signal to separate all 22 crop classes. The delta is -0.23% in accuracy and a slightly higher CV standard deviation. The correct deployment decision is to use the simpler baseline model to reduce preprocessing overhead on IoT edge devices.

---

## Step 5 — Model Diagnostics and Visualisations

| Chart | What it shows |
|---|---|
| Confusion matrix (22×22) | Near-perfect diagonal — jute, maize, mothbeans, lentil, blackgram, and rice are the only crops with any misclassification |
| Feature importance (horizontal bar) | 5 features above mean threshold: Rainfall, Humidity, K, P, NPK_total; mean line annotated in red |
| CV fold comparison (grouped bar) | All 3 models across all 10 folds with mean dashed lines — confirms stability, not just average performance |
| Per-class precision (horizontal bar) | Individual precision per crop sorted ascending; jute and maize are the only two below 1.0 |

---

## Step 6 — Live Prediction Function

`recommend_crop()` bridges the trained model and real-world deployment. It accepts 7 raw sensor readings, applies the same 4 feature engineering transformations used during training, and passes the resulting 11-feature vector through the trained Random Forest.

```python
result = recommend_crop(
    N=90, P=42, K=43,
    temperature=21, humidity=82,
    ph=6.5, rainfall=203
)
# Returns:
# {
#   'recommendation': 'rice',
#   'confidence': 91.5,
#   'top_3': {'rice': 91.5, 'jute': 8.5, 'blackgram': 0.0}
# }
```

The function returns the top crop, its confidence score as a percentage, and the full top-3 ranked alternatives. Returning confidence alongside the recommendation matters in practice: a 91% recommendation for rice is a strong signal, while a 40% recommendation should trigger a second review before committing to planting.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `N` | float | Nitrogen level (mg/kg) |
| `P` | float | Phosphorus level (mg/kg) |
| `K` | float | Potassium level (mg/kg) |
| `temperature` | float | Average temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `ph` | float | Soil pH (0–14) |
| `rainfall` | float | Annual rainfall (mm) |
| `model` | estimator | Trained sklearn classifier (default: Random Forest) |

---

## Key Findings

**Climate Outweighs Soil Chemistry** — Rainfall and Humidity are the two most important predictors, together accounting for a disproportionate share of the model's classification weight. Soil nutrients can be adjusted through fertilisation; ambient water availability cannot. This means climate zone is the primary constraint on crop viability, not soil chemistry.

**Feature Engineering Added No Value** — The baseline Random Forest on 7 raw sensors (99.55%) outperformed the engineered 11-feature model (99.32%). The sensor readings are already sufficiently discriminating without transformation. Adding composite features increased pipeline complexity with no accuracy gain and slightly higher CV variance. The right deployment choice is the simpler model.

**Potassium and Phosphorus Are Stronger Discriminators Than Nitrogen** — Despite Nitrogen's prominent role in plant growth science, Potassium (K) and Phosphorus (P) show greater inter-crop variability in this dataset, making them more useful signals for the classifier. Nitrogen fell below the mean importance threshold.

**The Dataset Is Exceptionally Well-Suited to Classification** — A 99.59% cross-validated accuracy with only 0.38% standard deviation on raw sensor data reflects how precisely IoT-derived environmental measurements separate crop profiles. This contrasts sharply with human behavioural data, where similar models typically achieve 70–85% accuracy with much higher variance.

**Jute and Maize Are the Only Ambiguous Crops** — Of 22 crop types, only jute (precision 0.95) and maize (precision 0.95) showed any misclassification. These two crops share overlapping sensor ranges with their nearest neighbours and represent the only cases where additional sensor data or a confidence threshold check would be warranted before acting on a recommendation.

---

## Business Recommendations

| Priority | Action | Driven by |
|---|---|---|
|  High | Deploy the 7-feature baseline Random Forest — not the engineered version — to IoT dashboards | Feature engineering finding |
|  High | Integrate `recommend_crop()` directly into farm management software with live sensor API feeds | Live prediction function |
|  High | Flag recommendations for jute and maize with a secondary confidence check before planting | Per-class precision analysis |
|  Medium | Set a minimum confidence threshold (e.g. 70%) below which the system escalates to an agronomist | Confidence scoring |
|  Medium | Collect additional sensor types (soil moisture, organic matter content) to close the jute/maize ambiguity | Misclassification analysis |
|  Low | Retrain annually on new sensor data as climate patterns shift — Rainfall and Humidity are the top drivers and are subject to long-term change | Feature importance |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

```bash
pip install -r requirements.txt
```
