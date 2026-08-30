# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that builds an interpretable fraud detection system for credit card transactions. The pipeline spans exploratory data analysis, feature engineering, multi-model benchmarking, hyperparameter tuning, decision-threshold calibration, and SHAP-based model interpretation — culminating in actionable business policies for a multi-tiered fraud decision architecture.

---

## Problem Statement

Credit card fraud is a heavily **imbalanced classification** problem. In this dataset, only **1.695%** of transactions (339 out of 20,000) are fraudulent — meaning a naïve baseline that always predicts "legitimate" achieves **98.3% accuracy** while catching **zero** fraud cases.

This project prioritizes **recall-oriented metrics** (PR-AUC, F₂-Score, Recall@85% Precision) to minimize missed fraud, and applies **cost-sensitive learning** and **decision threshold optimization** to balance detection rate against customer friction.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
- **Size:** 20,000 transactions × 26 features
- **Target:** `is_fraud` (binary: 0 = legitimate, 1 = fraudulent)
- **Class Distribution:** 98.3% legitimate / 1.7% fraudulent

### Key Features

| Category | Features |
| :--- | :--- |
| **Transaction** | `amount_usd`, `merchant_category`, `channel`, `time_of_day_hour`, `day_of_week` |
| **Card & Account** | `card_type`, `card_age_months`, `customer_age`, `account_balance_usd` |
| **Authentication** | `auth_method` (OTP, 3D Secure, Biometric, PIN, No Authentication) |
| **Behavioral Velocity** | `hours_since_last_txn`, `txn_count_last_24h`, `velocity_score` |
| **Device & Network** | `device_type`, `used_vpn`, `ip_country_mismatch`, `is_foreign_transaction` |
| **Risk Indicators** | `billing_shipping_mismatch`, `cvv_retry_count`, `is_new_merchant`, `is_ai_generated_scam_attempt`, `merchant_risk_score`, `prior_disputes` |

---

## Project Structure

```
credit-card-fraud-detection/
│
├── dataset/
│   └── dataset.csv                              # Raw transaction data (via Kaggle)
│
├── notebook/
│   ├── 01_importing_dataset.py                  # KaggleHub download script
│   ├── 02_exploratory_data_analysis.ipynb        # Deep-dive EDA & feature insights
│   ├── 03_modeling.ipynb                        # Multi-model benchmarking & evaluation
│   └── 04_tuning_and_interpretation.ipynb       # Tuning, threshold calibration, SHAP & export
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Pipeline Overview

```
 ┌──────────────────────┐
 │  1. Data Ingestion   │  Download from Kaggle via kagglehub
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  2. EDA & Profiling   │  Class imbalance, distributions, correlations, risk heatmaps
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  3. Feature Eng.     │  Log transforms, cumulative risk flags, ratio features
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  4. Model Selection  │  LightGBM, XGBoost, CatBoost, Logistic Regression, Balanced RF
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  5. Tuning &         │  GridSearchCV, F₂-optimized threshold calibration
 │     Calibration      │
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  6. Interpretation   │  SHAP beeswarm, waterfall plots, local case studies
 └────────┬─────────────┘
          ▼
 ┌──────────────────────┐
 │  7. Model Export &   │  Serialized champion model bundle, business policy design
 │     Recommendations  │
 └──────────────────────┘
```

---

## Exploratory Data Analysis

The EDA (`03_exploratory_data_analysis.ipynb`) investigates fraud patterns across six analytical dimensions:

### Key Findings

| Finding | Detail |
| :--- | :--- |
| **Class Imbalance** | Only **339 / 20,000** transactions (1.695%) are fraudulent |
| **Financial Dynamics** | Fraud transactions average **$181.25** vs **$131.58** for legitimate (+37.7%) |
| **Velocity Risk** | Fraud velocity scores average **28.97** vs **19.65** for legitimate (+47.4%) |
| **High-Risk Merchants** | Crypto Exchange (**4.99%**), Gift Cards (**4.38%**), Gaming (**3.67%**) fraud rates |
| **Authentication Gap** | `No Authentication` has **4.44%** fraud rate vs **0.78%** for `Biometric` |
| **AI Scam Flag** | `is_ai_generated_scam_attempt = True` → **9.07%** fraud rate (5.8× baseline) |
| **Compounding Risk** | Transactions with ≥3 risk flags show **16.5% fraud probability** (19× baseline) |

### Analysis Sections

1. **Environment Setup & Data Ingestion**
2. **Dataset Architecture & Data Integrity Checks**
3. **Target Variable Profiling** — Class imbalance quantification
4. **Financial & Transaction Amount Dynamics** — Distribution analysis, skewness
5. **Behavioral Velocity & Temporal Risk Patterns** — Velocity scores, CVV retries
6. **Merchant Profiling & Channel/Device Risk** — Category-level fraud rates
7. **Geopolitical, Network & AI-Generated Scam Vectors** — VPN, IP mismatch, AI scams
8. **Correlation Matrix & Statistical Significance Testing**
9. **Multivariate Interactions & Risk Heatmaps**

---

## Feature Engineering

Six engineered features were derived from EDA insights to strengthen signal extraction:

| Feature | Definition | Rationale |
| :--- | :--- | :--- |
| `log_amount_usd` | `log(1 + amount_usd)` | Normalize heavy right-skewed transaction amounts |
| `cumulative_risk_factor` | Sum of 6 binary risk flags | Quantify multi-factor risk compounding |
| `amount_to_balance_ratio` | `amount_usd / (account_balance_usd + 1)` | Measure relative financial strain on account |
| `velocity_to_gap_ratio` | `velocity_score / (hours_since_last_txn + 0.1)` | Detect sudden spending velocity spikes |
| `is_high_risk_merchant` | `merchant ∈ {Crypto, Gift Cards, Gaming}` | Flag high-liquidity merchant vulnerability |
| `is_unauthenticated` | `auth_method == "No Authentication"` | Flag missing step-up security |

---

## Modeling & Evaluation

Five classifiers were benchmarked in `04_modeling.ipynb` using **Stratified 5-Fold Cross-Validation** to preserve class distribution:

### Models Evaluated

| Model | Class Balancing Strategy |
| :--- | :--- |
| **LightGBM** | `scale_pos_weight ≈ 58` |
| **XGBoost** | `scale_pos_weight ≈ 58`, `eval_metric="aucpr"` |
| **CatBoost** | `auto_class_weights="Balanced"`, `eval_metric="PRAUC"` |
| **Logistic Regression** | `class_weight="balanced"` |
| **Balanced Random Forest** | Built-in per-class resampling |

### Evaluation Metrics

- **Primary:** PR-AUC (Average Precision) — robust under class imbalance
- **Secondary:** Recall@85% Precision, F₂-Score (recall-weighted)
- **Threshold Sweep:** Grid search over `[0.01, 1.00]` step 0.01

---

## Hyperparameter Tuning & Threshold Calibration

The tuning notebook (`05_tuning_and_interpretation.ipynb`) focuses on the **Logistic Regression** champion model:

### Tuning Strategy

- **GridSearchCV** (5-Fold Stratified) optimizing:
  - Regularization strength: *C* ∈ [0.001, 0.005, 0.01, 0.05, 0.077, 0.1, 0.5, 1.0, 5.0, 10.0]
  - Regularization type, solver selection

### Decision Threshold Calibration

The default threshold (*t* = 0.50) assumes symmetric costs. In fraud detection:
- **False Negatives** (missed fraud) are far more costly than **False Positives** (unnecessary verification)
- The optimal **F₂ threshold** was calibrated to *t*\* ≈ **0.08**, dramatically improving recall

### Comparison: Baseline vs Tuned

Both default threshold (*t* = 0.50) and calibrated optimal F₂ threshold (*t*\* ≈ 0.08) are compared side-by-side for Baseline vs Tuned Logistic Regression.

---

## Model Interpretation (SHAP)

SHAP (SHapley Additive exPlanations) analysis provides both global and local interpretability:

### Global Interpretation
- **Coefficient Analysis:** Odds ratios (*eᵝ*) quantify each feature's multiplicative effect on fraud probability
- **SHAP Beeswarm Plot:** Visualizes feature contribution distributions across all test transactions

### Local Case Studies (Transaction-Level Diagnostics)

Four operational cases examined with waterfall plots:

| Case | Scenario | Purpose |
| :--- | :--- | :--- |
| **Case 1** | True Positive (Caught Fraud) | Validate model correctly flags suspicious patterns |
| **Case 2** | True Negative (Cleared Legitimate) | Confirm model doesn't over-flag normal behavior |
| **Case 3** | False Positive | Understand sources of unnecessary customer friction |
| **Case 4** | False Negative (Missed Fraud) | Identify blind spots for model improvement |

---

## Business Recommendations

### Multi-Tiered Fraud Decision Architecture

| Risk Tier | Score Range | Action | Operational Impact |
| :--- | :--- | :--- | :--- |
| 🟢 **Tier 1: Green** | P̂ < 0.05 | Instant Auto-Approve | ~92.4% of legitimate volume cleared frictionlessly |
| 🟡 **Tier 2: Amber** | 0.05 ≤ P̂ < 0.35 | Dynamic 3DS / OTP Challenge | Intercepts ~80% of fraud with self-service resolution |
| 🔴 **Tier 3: Red** | P̂ ≥ 0.35 | Auto-Decline / Manual Review | High-precision fraud stop |

### Key Fraud Signals for Rule-Engine Hardening

1. **Cumulative Risk Multipliers:** ≥2 combined anomalies (VPN + IP mismatch + billing mismatch) multiply fraud odds by >2.4×
2. **Velocity Spike Detection:** Elevated `velocity_to_gap_ratio` indicates automated card-testing bot attacks
3. **High-Risk Merchants:** Crypto Exchange and Gift Cards require mandatory 3DS verification

### Model Governance

- **Real-Time Latency:** p99 inference < 5ms
- **Distribution Drift:** PSI alerts if > 0.10 on key features
- **Periodic Recalibration:** Monthly threshold adjustment for seasonal patterns

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, LightGBM, XGBoost, CatBoost |
| **Imbalanced Learning** | imbalanced-learn (BalancedRandomForest) |
| **Model Interpretation** | SHAP |
| **Data Acquisition** | kagglehub |

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost imbalanced-learn shap kagglehub
```

### Download Dataset

```bash
python notebook/01_importing_dataset.py
```

Or manually download from [Kaggle](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) and place `dataset.csv` in the `dataset/` directory.

### Run Notebooks

1. **EDA:** `notebook/03_exploratory_data_analysis.ipynb`
2. **Modeling:** `04_modeling.ipynb`
3. **Tuning & Interpretation:** `05_tuning_and_interpretation.ipynb`

---

## License

This project is open-source and available for educational and portfolio purposes.
