# Laptop Price Prediction Using Machine Learning

> Predicting laptop prices from real-world multi-source data using regression — Project 08

---

##  Project Overview

Laptop prices vary dramatically across brands, specifications, and regions — yet buyers and businesses often lack a systematic way to estimate fair market value. This project builds an end-to-end machine learning pipeline that predicts laptop prices from hardware specifications alone.

The core focus of this project is **data engineering** — sourcing, cleaning, and merging four real-world datasets before any modeling begins. The result is a merged master dataset of **3,918 rows** built from scratch, not a pre-cleaned Kaggle CSV.

---

##  Dataset Details

| Source | Rows | Currency |
|---|---|---|
| laptop_price01 | 885 | INR |
| laptop_price02 | 886 | INR |
| laptop_price03 | 1,270 | EUR → INR |
| laptop_price04 | 1,013 | INR |

**Final master dataset:** 3,918 rows × 10 columns · 0 nulls · 0 duplicates

---

##  Data Engineering Pipeline

Each dataset was cleaned independently before merging — following the **ETL (Extract → Transform → Load)** approach:

**Per-source cleaning:**
- Fixed encoding artifacts (`â`, superscript `ᵉ`) introduced during scraping
- Extracted numeric values from embedded strings (`'8GB DDR4'` → `ram_gb=8`, `ram_type='DDR4'`)
- Standardized OS values into 8 clean categories
- Dropped 23 corrupt rows with data-shifted columns (storage column containing GPU values)
- Corrected DDR6 → DDR5 (DDR6 does not exist as a consumer standard)
- Normalized `1000 GB` / `2000 GB` notation to proper TB-equivalent values in GB

**Schema merge:**
- Aligned 4 different column schemas to 9 common columns
- Converted EUR → INR for ds03
- Removed 135 exact duplicate rows post-concatenation

---

##  EDA — Key Findings

| # | Finding | Implication |
|---|---|---|
| 1 | Price skewness = 2.08 (right-skewed) | Applied `log1p(price)` as target variable |
| 2 | `ram_gb` correlation with price: r = 0.56 | Strongest numeric predictor |
| 3 | Brand drives 2–4× price premium (Razer vs AXL) | Target-encoded to capture brand tier |
| 4 | SSD vs HDD creates a larger gap than doubling storage size | Storage type > storage size |
| 5 | macOS appears exclusively on Apple laptops | OS and brand are correlated — multicollinearity risk |
| 6 | IQR outliers are genuine premium products | Kept all outliers; log-transform handles their leverage |

---

##  Feature Engineering

Transformed 10 raw columns into **44 model-ready features**:

| New Feature | Source | Values |
|---|---|---|
| `cpu_brand` | processor string | Intel / AMD / Apple / Qualcomm / MediaTek |
| `cpu_tier` | processor string | i3 / i5 / i7 / i9 / Ryzen 3–9 / Apple Silicon |
| `gpu_type` | gpu string | Dedicated / Integrated |
| `brand_encoded` | brand (target encoding) | Median log-price per brand |
| `price_log` | price | `log1p(price)` — skewness reduced from 2.08 → ~0 |

Categorical encoding strategy:
- `brand` → **Target encoding** (avoids 36 sparse dummy columns)
- `os`, `storage_type`, `ram_type`, `gpu_type`, `cpu_brand`, `cpu_tier` → **One-hot encoding**

---

##  Model Results

All models trained on scaled features (StandardScaler) with an 80/20 train/test split.

| Model | R² | MAE (log) | RMSE (log) | MAE (INR) |
|---|---|---|---|---|
| Linear Regression | 0.8098 | 0.2160 | 0.2863 | ₹23,557 |
| **Ridge Regression** **(Winner)** | **0.8102** | **0.2160** | **0.2860** | **₹23,560** |
| Lasso Regression | 0.7545 | 0.2479 | 0.3253 | ₹27,069 |

**Best model: Ridge Regression (α=1.0)**

Ridge outperforms Linear Regression marginally (R² difference < 0.001), confirming the data is not severely overfitting. Lasso underperformed because L1 regularization eliminated features that were still contributing genuine signal.

The average prediction error of **₹23,557** on a median laptop price of **₹74,390** is expected given that key features such as display size, battery life, and weight were not consistently available across all four source datasets.

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Notebook | Jupyter Notebook |

---
##  Repository Structure

```
Laptop-Price-Prediction-Model/
│
├── datasets/
│   ├── laptop_price01_clean.csv
│   ├── laptop_price02_clean.csv
│   ├── laptop_price03_clean.csv
│   ├── laptop_price04_clean.csv
│   └── laptop_master.csv               ← Final merged dataset (3,918 rows)
│
├── 01_EDA.ipynb ← Standalone EDA notebook (EDA)
├── 02Feature_Engineering_&_Modelling.ipynb ← Main notebook  (Feature Engineering + Modeling)
├── laptop_price_presentation.pptx ← Project presentation deck
└── README.md
```
---

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OfficialSamarthsingh/Laptop-Price-Prediction-Model.git
cd Laptop-Price-Prediction-Model

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Open the main notebook
jupyter notebook laptop_price_prediction.ipynb
```

Run all cells top to bottom — the notebook covers EDA, feature engineering, and modeling end to end.

---

##  Project Highlights

-  Built a full ETL pipeline per source before merging
-  Extracted structured features from 400+ unique processor and GPU strings
-  Applied target encoding for brand to avoid dimensionality explosion
-  R² = 0.81 on real-world merged data without display size, battery, or weight features
-  Model saved with Joblib for production use without retraining

---

##  Author

**Samarth Singh**
Aspiring Data Analyst · SRMU, Lucknow (B.Tech CSE, 2026)

[![GitHub](https://img.shields.io/badge/GitHub-OfficialSamarthsingh-181717?logo=github)](https://github.com/OfficialSamarthsingh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-samarth--singh-0A66C2?logo=linkedin)](https://linkedin.com/in/samarthsinghofficial)
[![Portfolio](https://img.shields.io/badge/Portfolio-samarthsinghportfolio.netlify.app-4A90D9?logo=netlify)](https://samarthsinghportfolio.netlify.app)

---

*Part of my data analytics portfolio — building real-world projects to transition into a Data Analyst role.*
