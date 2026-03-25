import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.diagnostics import (
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_residuals,
)
from src.preprocessing import preprocess, split_data


def test_plot_residuals_returns_4_panel_figure():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    fig = plot_residuals(y_true, y_pred, "Test")
    assert fig is not None
    assert len(fig.axes) == 4  # pred vs actual, dist, homoscedasticity, Q-Q


def test_plot_learning_curves_returns_figure(clean_df: pd.DataFrame):
    X_train, X_test, y_train, _ = split_data(clean_df)
    X_train_proc, _, _ = preprocess(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    fig = plot_learning_curves(LinearRegression(), X_train_proc, y_train, "LR")
    assert fig is not None


def test_plot_feature_importance_returns_figure():
    imp_df = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance_mean": [0.5, 0.3, 0.1],
            "importance_std": [0.05, 0.03, 0.01],
        }
    )
    fig = plot_feature_importance(imp_df, "Test")
    assert fig is not None


def test_plot_model_comparison_with_adj_r2():
    mock_results = {
        "Model_A": pd.DataFrame(
            {"Train": {"R2": 0.9, "Adj_R2": 0.89}, "Test": {"R2": 0.8, "Adj_R2": 0.79}}
        ),
        "Model_B": pd.DataFrame(
            {"Train": {"R2": 0.85, "Adj_R2": 0.84}, "Test": {"R2": 0.83, "Adj_R2": 0.82}}
        ),
    }
    fig = plot_model_comparison(mock_results)
    assert fig is not None
