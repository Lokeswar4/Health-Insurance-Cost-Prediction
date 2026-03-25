import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.model import (
    adjusted_r2,
    cross_validate,
    evaluate,
    train_and_evaluate,
    tune_gradient_boosting,
    tune_lightgbm,
    tune_xgboost,
)
from src.preprocessing import add_interaction_features, preprocess, split_data


def test_evaluate_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    metrics = evaluate(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


def test_evaluate_returns_all_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    metrics = evaluate(y_true, y_pred)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "R2"}


def test_adjusted_r2_penalizes_features():
    r2 = 0.85
    adj_fewer = adjusted_r2(r2, n=100, p=3)
    adj_more = adjusted_r2(r2, n=100, p=20)
    assert adj_fewer > adj_more


def test_train_and_evaluate_includes_adj_r2(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    _, results = train_and_evaluate(LinearRegression(), X_train_proc, y_train, X_test_proc, y_test)
    assert "Adj_R2" in results.index
    assert results.loc["Adj_R2", "Test"] <= results.loc["R2", "Test"]


def test_train_and_evaluate_returns_model_and_results(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model, results = train_and_evaluate(
        LinearRegression(), X_train_proc, y_train, X_test_proc, y_test
    )
    assert hasattr(model, "predict")
    assert isinstance(results, pd.DataFrame)
    assert "Train" in results.columns
    assert "Test" in results.columns


def test_linear_regression_r2_above_baseline(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    _, results = train_and_evaluate(LinearRegression(), X_train_proc, y_train, X_test_proc, y_test)
    assert results.loc["R2", "Test"] > 0.5


def test_interaction_features_improve_lr(clean_df: pd.DataFrame):
    """LR with interaction features should outperform baseline LR."""
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    X_train_eng, X_test_eng = add_interaction_features(X_train_proc, X_test_proc)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    _, base_res = train_and_evaluate(LinearRegression(), X_train_proc, y_train, X_test_proc, y_test)
    _, eng_res = train_and_evaluate(LinearRegression(), X_train_eng, y_train, X_test_eng, y_test)
    assert eng_res.loc["R2", "Test"] > base_res.loc["R2", "Test"]


def test_cross_validate_returns_scores(clean_df: pd.DataFrame):
    X_train, X_test, y_train, _ = split_data(clean_df)
    X_train_proc, _, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)

    scores = cross_validate(LinearRegression(), X_train_proc, y_train, cv=3)
    assert len(scores) == 3
    assert all(s > 0 for s in scores)


def test_tune_gradient_boosting_returns_fitted_model(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)

    model = tune_gradient_boosting(X_train_proc, y_train, n_trials=3)
    assert hasattr(model, "predict")
    preds = model.predict(X_test_proc)
    assert len(preds) == len(y_test)


def test_tune_xgboost_r2_above_baseline(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = tune_xgboost(X_train_proc, y_train, n_trials=3)
    _, results = train_and_evaluate(model, X_train_proc, y_train, X_test_proc, y_test, "XGBoost")
    assert results.loc["R2", "Test"] > 0.5


def test_tune_lightgbm_r2_above_baseline(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = tune_lightgbm(X_train_proc, y_train, n_trials=3)
    _, results = train_and_evaluate(
        model, X_train_proc, y_train, X_test_proc, y_test, "LightGBM"
    )
    assert results.loc["R2", "Test"] > 0.5


def test_tune_xgboost_has_adj_r2(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = tune_xgboost(X_train_proc, y_train, n_trials=3)
    _, results = train_and_evaluate(model, X_train_proc, y_train, X_test_proc, y_test, "XGBoost")
    assert "Adj_R2" in results.index
    assert results.loc["Adj_R2", "Test"] <= results.loc["R2", "Test"]
