# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML project predicting medical insurance costs from patient demographics (Kaggle dataset, 1,338 rows, 7 features). Key insight: 3 distinct sub-populations driven by smoker x BMI interaction. 7 models across 4 phases; best models are tuned tree ensembles (GB/XGBoost/LightGBM) via Optuna Bayesian search.

## Commands

```bash
uv sync --dev                                   # install all deps
uv run python main.py                           # full pipeline (EDA + modeling + save model)
uv run python main.py --model-only              # modeling only
uv run python main.py --save-plots              # save 16 plots to outputs/ (5 EDA + 11 diagnostics)
uv run python predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker yes --region northeast
uv run python predict.py --csv data/insurance.csv --output predictions.csv
uv run pytest tests/ -v                         # run all 40 tests
uv run pytest tests/test_model.py -v            # single test file
uv run pytest tests/ -k "test_vif"              # pattern match
uv run ruff check src/ tests/ main.py predict.py
uv run ruff format src/ tests/ main.py predict.py
```

## Architecture

```
main.py                 # CLI entry point (--model-only, --save-plots)
predict.py              # Prediction CLI (--age/--sex/... or --csv)
src/
  config.py             # Paths, feature lists, hyperparameters, MODEL_DIR
  data_loader.py        # Load CSV, drop duplicates
  eda.py                # Target analysis, interactions, VIF, hypothesis tests, plots
  preprocessing.py      # Stratified split, ColumnTransformer, interaction features
  model.py              # 7 models, Optuna Bayesian tuning, bootstrap CIs, permutation importance, Adj R²
  diagnostics.py        # 4-panel residuals (incl Q-Q), learning curves, importance bars, comparison
  persistence.py        # FullPipeline + PredictionPipeline, joblib save/load
tests/
  conftest.py           # Shared fixtures (raw_df, clean_df)
  test_data_loader.py   # Data loading and dedup
  test_eda.py           # Outliers, hypothesis tests, VIF, effect sizes
  test_preprocessing.py # Pipeline shape/dtype/scaling, interaction features, leakage checks
  test_model.py         # Metrics, Adj R², interaction improvement assertion, CV, Optuna tuning
  test_diagnostics.py   # 4-panel residuals, importance charts, comparison chart
  test_persistence.py   # FullPipeline, PredictionPipeline roundtrip, save/load
```

### Data Flow
1. `data_loader.load_data()` -> raw DataFrame (deduped)
2. `preprocessing.split_data()` -> stratified 80/20 train/test
3. `preprocessing.preprocess()` -> ColumnTransformer -> 9 base features
4. `preprocessing.add_interaction_features()` -> 4 engineered features (13 total, thresholds from train)
5. `model.run_all_models()` -> 7 models in 4 phases, returns results + fitted models + importance DFs
6. `persistence.save_model()` -> joblib dump (FullPipeline + model)
7. `predict.py` -> loads saved pipeline, predicts from CLI args or CSV

### Models (in order)
Phase 1 — Baseline:
1. LR baseline (9 features) -> R²=0.750, Adj R²=0.741

Phase 2 — Feature-Engineered Linear Models:
2. LR + interactions (13 features) -> R²=0.852, Adj R²=0.844
3. Ridge + interactions (L2) -> R²=0.853, Adj R²=0.845
4. Lasso + interactions (L1) -> R²=0.853, Adj R²=0.845

Phase 3 — Ensemble Models with Bayesian Tuning (Optuna TPE, 30 trials each):
5. Gradient Boosting (tuned) -> R²≈0.864
6. XGBoost (tuned) -> R²≈0.87+
7. LightGBM (tuned) -> R²≈0.87+

Phase 4 — Permutation importance for LR+interactions, GB, XGBoost, LightGBM

### Tuning
- Replaced GridSearchCV (27 fixed combos) with **Optuna TPE** (Bayesian, 30 trials)
- `OPTUNA_TRIALS = 30` in `src/config.py` (override with `n_trials` param)
- Search spaces: n_estimators [100-500], learning_rate [0.01-0.3 log], subsample [0.5-1.0],
  max_depth [2-6/8], plus XGB/LGBM-specific: colsample_bytree, reg_alpha, reg_lambda, num_leaves

## Tooling
- **uv** -- dependency management and venv
- **ruff** -- linting (E, F, I, UP, B, SIM) and formatting
- **pytest** -- 40 tests including VIF, leakage, save/load roundtrip, Optuna tuning
- **GitHub Actions** -- CI on Python 3.12 and 3.13
- **joblib** -- model persistence
- **statsmodels** -- VIF computation
- **optuna** -- Bayesian hyperparameter search (TPE sampler)
- **xgboost** -- gradient boosted trees (XGBRegressor)
- **lightgbm** -- gradient boosted trees (LGBMRegressor)
