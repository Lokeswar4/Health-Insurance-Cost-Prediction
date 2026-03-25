# Health Insurance Cost Prediction

[![CI](https://github.com/Lokeswar4/Health-Insurance-Cost-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Lokeswar4/Health-Insurance-Cost-Prediction/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Predicting medical insurance charges from patient demographics using the [Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (1,338 rows, 7 features). Full ML workflow: statistical EDA, domain-driven feature engineering, regularized regression, gradient boosted ensembles with Bayesian hyperparameter search, residual diagnostics, and a production-ready prediction CLI.

---

## Results

| Model | Test R² | Adj R² | Test MAE | CV R² (5-fold) |
|-------|---------|--------|----------|----------------|
| Linear Regression (baseline) | 0.750 | 0.741 | $4,086 | 0.743 ± 0.017 |
| Linear Regression + interactions | 0.852 | 0.844 | $2,582 | 0.845 ± 0.012 |
| Ridge Regression + interactions | 0.853 | 0.845 | $2,549 | 0.844 ± 0.013 |
| Lasso Regression + interactions | 0.853 | 0.845 | $2,565 | 0.845 ± 0.012 |
| **Gradient Boosting** (Optuna-tuned) | **0.864** | **0.859** | **$2,395** | **0.857 ± 0.017** |
| **XGBoost** (Optuna-tuned) | — | — | — | — |
| **LightGBM** (Optuna-tuned) | — | — | — | — |

> XGBoost and LightGBM results vary per Optuna run (30 Bayesian trials each). Run `python main.py` to see current numbers.

Bootstrap 95% CI for test R² (1,000 resamples): LR+interactions [0.776, 0.912] · GB [0.789, 0.925]

---

## Key Findings

- The data contains **3 distinct cost sub-populations**: non-smokers (low cost, age-driven), smokers with BMI < 30 (moderate), and smokers with BMI ≥ 30 (very high)
- The **smoker × BMI interaction** is the single most predictive feature (permutation importance = 0.472) — adding it lifts LR from R²=0.75 → 0.85
- Smokers pay **3.8× more** on average (Cohen's d = 2.57, p < 10⁻¹⁰⁰, Mann-Whitney U confirmed)
- VIF analysis flags multicollinearity between age/age² and smoker/smoker_x_bmi, justifying Ridge regularization
- **Feature engineering > model complexity**: LR + 4 interaction features matches tuned Ridge/Lasso
- Ensemble models (GB, XGBoost, LightGBM) capture non-linear residual structure missed by linear models

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Lokeswar4/Health-Insurance-Cost-Prediction.git
cd Health-Insurance-Cost-Prediction
uv sync --dev          # creates .venv and installs all dependencies
```

---

## Usage

### Run the full pipeline

```bash
# EDA + all models + save model (default)
uv run python main.py

# Skip EDA, run modeling only
uv run python main.py --model-only

# Save all 16 plots to outputs/
uv run python main.py --save-plots

# Adjust log verbosity
uv run python main.py --log-level DEBUG
```

### Predict insurance costs

```bash
# Single prediction
uv run python predict.py \
  --age 35 --sex male --bmi 28.5 \
  --children 2 --smoker yes --region northeast
# -> Predicted insurance cost: $25,658.67

# Batch prediction from CSV
uv run python predict.py \
  --csv data/insurance.csv \
  --output predictions.csv
```

### Development

```bash
make install   # uv sync --dev
make test      # pytest tests/ -v  (40 tests)
make lint      # ruff check
make format    # ruff format
make run       # python main.py
```

---

## Methodology

### 1. Exploratory Data Analysis

- Target distribution: skewness=1.5, CV=0.91 — right-skewed, driven by smoker sub-population
- Sub-population identification: 3 clusters visible in charges vs BMI scatter (colored by smoker)
- Interaction analysis: smoker × BMI, smoker × age, children × smoker pivot tables
- Hypothesis testing with effect sizes:
  - Welch's t-test (unequal variance): smoker vs non-smoker charges
  - Mann-Whitney U (non-parametric confirmation)
  - Effect sizes: Cohen's d = 2.57, rank-biserial r = 0.93
- Pearson and Spearman correlation heatmaps
- VIF (Variance Inflation Factor) analysis — multicollinearity flagged at VIF > 5

### 2. Feature Engineering

Four interaction features computed from **training data only** (no data leakage):

| Feature | Description | Permutation Importance |
|---------|-------------|----------------------|
| `smoker_x_bmi` | smoker flag × standardized BMI | **0.472** |
| `smoker_x_age` | smoker flag × standardized age | 0.089 |
| `age_squared` | non-linear age effect on premiums | 0.041 |
| `obese_smoker` | binary: smoker AND BMI ≥ 30 (threshold from training 55th percentile) | 0.028 |

### 3. Preprocessing

- Stratified 80/20 train/test split (stratified on smoker × sex × region)
- sklearn `ColumnTransformer`:
  - `MinMaxScaler` — numerical features (age, bmi, children)
  - `OrdinalEncoder` — binary categoricals (sex, smoker)
  - `OneHotEncoder` — nominal categoricals (region)

### 4. Modeling

Four phases:

**Phase 1 — Baseline**
- Linear Regression on 9 original features: establishes R²=0.750 floor

**Phase 2 — Feature-Engineered Linear Models**
- Linear Regression + 4 interaction features (+10pp R² gain)
- Ridge Regression (L2): addresses multicollinearity flagged by VIF analysis
- Lasso Regression (L1): feature selection — retains all 13 features (none zeroed out)

**Phase 3 — Ensemble Models with Bayesian Tuning (Optuna)**

All three ensemble models are tuned with **Optuna TPE** (Tree-structured Parzen Estimator), a Bayesian method that learns from prior trials to focus on promising hyperparameter regions. Replaces GridSearchCV which exhausted a fixed 27-combination grid.

| Model | Search Space |
|-------|-------------|
| Gradient Boosting | n_estimators [100–500], max_depth [2–6], learning_rate [0.01–0.3 log], subsample [0.5–1.0], min_samples_split [2–20] |
| XGBoost | + colsample_bytree [0.5–1.0], reg_alpha [0–10], reg_lambda [0–10], max_depth [2–8] |
| LightGBM | + num_leaves [20–150] instead of max_depth |

Default: 30 Optuna trials per model (configure via `OPTUNA_TRIALS` in `src/config.py`).

**Phase 4 — Feature Importance**
- Permutation importance on test set (30 repeats) for LR+interactions, GB, XGBoost, LightGBM

### 5. Evaluation and Diagnostics

- Metrics: MAE, RMSE, MAPE, R², **Adjusted R²** (penalizes feature count)
- 5-fold cross-validation with automatic overfitting detection (gap > 0.05 triggers warning)
- Bootstrap 95% confidence intervals for test R² (1,000 resamples)
- Residual diagnostics: predicted vs actual, residual distribution, homoscedasticity check, **Q-Q plot** (normality)
- Learning curves (bias-variance tradeoff, 10 training set sizes)

### 6. Model Persistence and Prediction

- Best model saved via joblib as a `PredictionPipeline` (raw features → prediction)
- `FullPipeline` is an sklearn-compatible `BaseEstimator`+`TransformerMixin` that handles preprocessing + interaction feature thresholds from training
- `predict.py` supports both single inference (CLI flags) and batch inference (`--csv` → CSV output)

---

## Generated Plots (`--save-plots`)

### EDA (5 plots)

| File | Description |
|------|-------------|
| `distributions.png` | Histograms of numerical features and charges |
| `boxplots.png` | Box plots by smoker/sex/region |
| `correlation_spearman.png` | Spearman rank correlation heatmap |
| `correlation_pearson.png` | Pearson correlation heatmap |
| `smoker_scatter.png` | Charges vs BMI/age coloured by smoker status — 3 sub-populations |

### Diagnostics (11 plots)

| File | Description |
|------|-------------|
| `residuals_lr.png` | LR 4-panel residual diagnostics (incl. Q-Q plot) |
| `residuals_gb.png` | Gradient Boosting 4-panel residual diagnostics |
| `residuals_xgb.png` | XGBoost 4-panel residual diagnostics |
| `residuals_lgbm.png` | LightGBM 4-panel residual diagnostics |
| `learning_curve_lr.png` | LR bias-variance learning curve |
| `learning_curve_gb.png` | GB bias-variance learning curve |
| `feature_importance_lr.png` | LR permutation importance bar chart |
| `feature_importance_gb.png` | GB permutation importance bar chart |
| `feature_importance_xgb.png` | XGBoost permutation importance bar chart |
| `feature_importance_lgbm.png` | LightGBM permutation importance bar chart |
| `model_comparison.png` | R² vs Adjusted R² for all 7 models |

---

## Project Structure

```
├── main.py                     # Pipeline entry point (EDA + modeling + save)
├── predict.py                  # Prediction CLI — single and batch inference
├── src/
│   ├── config.py               # Paths, RANDOM_STATE, OPTUNA_TRIALS, feature lists
│   ├── data_loader.py          # Load CSV, drop duplicates
│   ├── eda.py                  # Target analysis, VIF, hypothesis tests, plots
│   ├── preprocessing.py        # Stratified split, ColumnTransformer, interaction features
│   ├── model.py                # 7 models, Optuna tuning, CV, bootstrap CIs, importance
│   ├── diagnostics.py          # 4-panel residuals, Q-Q, learning curves, comparison charts
│   └── persistence.py          # FullPipeline + PredictionPipeline, joblib save/load
├── tests/                      # pytest — 40 tests
│   ├── conftest.py             # Shared fixtures (raw_df, clean_df)
│   ├── test_data_loader.py     # Loading and deduplication
│   ├── test_eda.py             # Outliers, hypothesis tests, VIF, effect sizes
│   ├── test_preprocessing.py   # Pipeline shape/dtype, scaling, leakage checks
│   ├── test_model.py           # Metrics, CV, Optuna tuning (XGB, LGBM, GB)
│   ├── test_diagnostics.py     # Plot output shapes and types
│   └── test_persistence.py     # FullPipeline roundtrip, save/load
├── data/
│   └── insurance.csv           # Kaggle Medical Cost Personal Dataset
├── pyproject.toml              # Dependencies: pandas, scikit-learn, xgboost, lightgbm, optuna, …
├── Makefile                    # install / test / lint / format / run
├── CLAUDE.md                   # AI assistant context (architecture, commands, design decisions)
└── .github/workflows/ci.yml    # CI: ruff lint + pytest on Python 3.12 and 3.13
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, linear models, GB, evaluation |
| `xgboost` | XGBRegressor — gradient boosted trees |
| `lightgbm` | LGBMRegressor — fast gradient boosted trees |
| `optuna` | Bayesian hyperparameter search (TPE sampler) |
| `scipy` | Q-Q plots, Mann-Whitney U test |
| `statsmodels` | VIF computation |
| `matplotlib`, `seaborn` | Visualisations |
| `joblib` | Model persistence |

---

## License

Apache 2.0
