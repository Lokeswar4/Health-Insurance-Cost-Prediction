import logging

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from src.config import OPTUNA_TRIALS, RANDOM_STATE

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── evaluation helpers ────────────────────────────────────────────────────────


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, R²."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Compute adjusted R² given R², sample size n, and number of predictors p."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, name: str = "Model"):
    """Train a model, print train/test metrics with adjusted R², return fitted model."""
    logger.info("model.fit name=%s samples=%d features=%d", name, len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = evaluate(y_train, train_preds)
    test_metrics = evaluate(y_test, test_preds)

    n_train, p = X_train.shape
    n_test = X_test.shape[0]
    train_metrics["Adj_R2"] = adjusted_r2(train_metrics["R2"], n_train, p)
    test_metrics["Adj_R2"] = adjusted_r2(test_metrics["R2"], n_test, p)

    results = pd.DataFrame({"Train": train_metrics, "Test": test_metrics}).round(3)
    print(f"\n{'=' * 55}")
    print(f" {name}")
    print(f"{'=' * 55}")
    print(results.to_string())

    overfit = train_metrics["R2"] - test_metrics["R2"]
    if overfit > 0.05:
        logger.warning(
            "model.overfit name=%s train_r2=%.3f test_r2=%.3f gap=%.3f",
            name,
            train_metrics["R2"],
            test_metrics["R2"],
            overfit,
        )
        print(f"  Warning: Overfit gap: {overfit:.3f} (train R2 - test R2)")

    return model, results


def cross_validate(model, X_train, y_train, cv: int = 5) -> np.ndarray:
    """K-fold cross-validation returning mean and std of R² scores."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    print(f"  CV R2 scores: {scores.round(3)}")
    print(f"  CV R2 mean:   {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores


def bootstrap_test_r2(model, X_test, y_test, n_boot: int = 1000) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for test R²."""
    rng = np.random.default_rng(RANDOM_STATE)
    preds = model.predict(X_test)
    n = len(y_test)
    r2_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r2_scores[i] = r2_score(np.asarray(y_test)[idx], preds[idx])
    lo, hi = np.percentile(r2_scores, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for test R2: [{lo:.3f}, {hi:.3f}]")
    return lo, hi


# ── optuna objectives ─────────────────────────────────────────────────────────


def _gb_objective(trial, X_train, y_train) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }
    model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()


def _xgb_objective(trial, X_train, y_train) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }
    model = XGBRegressor(**params, random_state=RANDOM_STATE, verbosity=0)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()


def _lgbm_objective(trial, X_train, y_train) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }
    model = LGBMRegressor(**params, random_state=RANDOM_STATE, verbose=-1)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()


# ── tuning functions ──────────────────────────────────────────────────────────


def _run_optuna(objective_fn, X_train, y_train, n_trials: int, name: str):
    """Run an Optuna TPE study and return best params + best CV R²."""
    logger.info("tuning.start model=%s trials=%d cv_folds=5", name, n_trials)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(lambda trial: objective_fn(trial, X_train, y_train), n_trials=n_trials)
    logger.info(
        "tuning.complete model=%s best_cv_r2=%.3f best_params=%s",
        name,
        study.best_value,
        study.best_params,
    )
    print(f"\n  Best CV R2: {study.best_value:.3f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def tune_gradient_boosting(X_train, y_train, n_trials: int = OPTUNA_TRIALS):
    """Bayesian hyperparameter tuning for Gradient Boosting via Optuna TPE."""
    best = _run_optuna(_gb_objective, X_train, y_train, n_trials, "GradientBoosting")
    model = GradientBoostingRegressor(**best, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def tune_xgboost(X_train, y_train, n_trials: int = OPTUNA_TRIALS):
    """Bayesian hyperparameter tuning for XGBoost via Optuna TPE."""
    best = _run_optuna(_xgb_objective, X_train, y_train, n_trials, "XGBoost")
    model = XGBRegressor(**best, random_state=RANDOM_STATE, verbosity=0)
    model.fit(X_train, y_train)
    return model


def tune_lightgbm(X_train, y_train, n_trials: int = OPTUNA_TRIALS):
    """Bayesian hyperparameter tuning for LightGBM via Optuna TPE."""
    best = _run_optuna(_lgbm_objective, X_train, y_train, n_trials, "LightGBM")
    model = LGBMRegressor(**best, random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train, y_train)
    return model


# ── permutation importance ────────────────────────────────────────────────────


def compute_permutation_importance(model, X_test, y_test, name: str = "Model") -> pd.DataFrame:
    """Compute and print permutation importance on test set."""
    logger.info("importance.permutation model=%s repeats=30 scoring=r2", name)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="r2"
    )
    imp_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print(f"\n  Permutation Importance ({name}):")
    for _, row in imp_df.iterrows():
        bar = "#" * int(row["importance_mean"] * 100)
        print(
            f"    {row['feature']:<16} "
            f"{row['importance_mean']:>6.3f} +/- {row['importance_std']:.3f}  {bar}"
        )
    return imp_df


# ── main pipeline ─────────────────────────────────────────────────────────────


def run_all_models(X_train, y_train, X_test, y_test, X_train_eng, X_test_eng):
    """Train all models and compare results."""
    all_results = {}
    fitted_models = {}

    # Phase 1: Baseline
    print("\n" + "-" * 55)
    print(" PHASE 1: Baseline Models (original features)")
    print("-" * 55)

    lr, lr_res = train_and_evaluate(
        LinearRegression(),
        X_train,
        y_train,
        X_test,
        y_test,
        name="Linear Regression (baseline)",
    )
    print("  5-fold CV:")
    cross_validate(LinearRegression(), X_train, y_train)
    all_results["LR_baseline"] = lr_res
    fitted_models["LR_baseline"] = (lr, X_test)

    # Phase 2: Feature-engineered linear models
    print("\n" + "-" * 55)
    print(" PHASE 2: Feature-Engineered Linear Models")
    print("-" * 55)

    lr_eng, lr_eng_res = train_and_evaluate(
        LinearRegression(),
        X_train_eng,
        y_train,
        X_test_eng,
        y_test,
        name="Linear Regression + interactions",
    )
    print("  5-fold CV:")
    cross_validate(LinearRegression(), X_train_eng, y_train)
    bootstrap_test_r2(lr_eng, X_test_eng, y_test)
    all_results["LR_interactions"] = lr_eng_res
    fitted_models["LR_interactions"] = (lr_eng, X_test_eng)

    ridge, ridge_res = train_and_evaluate(
        Ridge(alpha=1.0),
        X_train_eng,
        y_train,
        X_test_eng,
        y_test,
        name="Ridge Regression + interactions",
    )
    print("  5-fold CV:")
    cross_validate(Ridge(alpha=1.0), X_train_eng, y_train)
    all_results["Ridge_interactions"] = ridge_res

    lasso, lasso_res = train_and_evaluate(
        Lasso(alpha=10.0, max_iter=10000),
        X_train_eng,
        y_train,
        X_test_eng,
        y_test,
        name="Lasso Regression + interactions",
    )
    print("  5-fold CV:")
    cross_validate(Lasso(alpha=10.0, max_iter=10000), X_train_eng, y_train)
    lasso_coefs = pd.Series(lasso.coef_, index=X_train_eng.columns)
    kept = lasso_coefs[lasso_coefs.abs() > 0.01]
    dropped = lasso_coefs[lasso_coefs.abs() <= 0.01]
    print(f"  Lasso kept {len(kept)} features, dropped {len(dropped)}: {list(dropped.index)}")
    all_results["Lasso_interactions"] = lasso_res

    # Phase 3: Ensemble models with Bayesian tuning (Optuna TPE)
    print("\n" + "-" * 55)
    print(" PHASE 3: Ensemble Models with Bayesian Tuning (Optuna)")
    print("-" * 55)

    print(f"\n  Tuning Gradient Boosting ({OPTUNA_TRIALS} trials)...")
    best_gb = tune_gradient_boosting(X_train, y_train)
    _, gb_res = train_and_evaluate(
        best_gb, X_train, y_train, X_test, y_test, name="Gradient Boosting (tuned)"
    )
    print("  5-fold CV:")
    cross_validate(best_gb, X_train, y_train)
    bootstrap_test_r2(best_gb, X_test, y_test)
    all_results["GradientBoosting_tuned"] = gb_res
    fitted_models["GradientBoosting_tuned"] = (best_gb, X_test)

    print(f"\n  Tuning XGBoost ({OPTUNA_TRIALS} trials)...")
    best_xgb = tune_xgboost(X_train, y_train)
    _, xgb_res = train_and_evaluate(
        best_xgb, X_train, y_train, X_test, y_test, name="XGBoost (tuned)"
    )
    print("  5-fold CV:")
    cross_validate(best_xgb, X_train, y_train)
    bootstrap_test_r2(best_xgb, X_test, y_test)
    all_results["XGBoost_tuned"] = xgb_res
    fitted_models["XGBoost_tuned"] = (best_xgb, X_test)

    print(f"\n  Tuning LightGBM ({OPTUNA_TRIALS} trials)...")
    best_lgbm = tune_lightgbm(X_train, y_train)
    _, lgbm_res = train_and_evaluate(
        best_lgbm, X_train, y_train, X_test, y_test, name="LightGBM (tuned)"
    )
    print("  5-fold CV:")
    cross_validate(best_lgbm, X_train, y_train)
    bootstrap_test_r2(best_lgbm, X_test, y_test)
    all_results["LightGBM_tuned"] = lgbm_res
    fitted_models["LightGBM_tuned"] = (best_lgbm, X_test)

    # Phase 4: Feature importance
    print("\n" + "-" * 55)
    print(" PHASE 4: Feature Importance Analysis")
    print("-" * 55)

    lr_imp = compute_permutation_importance(lr_eng, X_test_eng, y_test, "LR + interactions")
    gb_imp = compute_permutation_importance(best_gb, X_test, y_test, "Gradient Boosting")
    xgb_imp = compute_permutation_importance(best_xgb, X_test, y_test, "XGBoost")
    lgbm_imp = compute_permutation_importance(best_lgbm, X_test, y_test, "LightGBM")

    # Final comparison
    print("\n" + "=" * 55)
    print(" FINAL MODEL COMPARISON")
    print("=" * 55)
    comparison = pd.DataFrame({name: res["Test"] for name, res in all_results.items()}).T
    comparison = comparison.sort_values("R2", ascending=False)
    print(comparison.round(3).to_string())

    imp_dfs = {"LR": lr_imp, "GB": gb_imp, "XGB": xgb_imp, "LGBM": lgbm_imp}
    return all_results, fitted_models, imp_dfs
