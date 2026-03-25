import pandas as pd

from src.eda import (
    analyze_interactions,
    analyze_target,
    check_outliers,
    compute_vif,
    plot_correlation_heatmap,
    plot_smoker_scatter,
    smoker_hypothesis_test,
)
from src.preprocessing import add_interaction_features, preprocess, split_data


def test_check_outliers_no_outliers():
    s = pd.Series([1, 2, 3, 4, 5])
    n_out, lower, upper = check_outliers(s)
    assert n_out == 0
    assert lower < 1
    assert upper > 5


def test_check_outliers_with_outlier():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    n_out, _, _ = check_outliers(s)
    assert n_out >= 1


def test_smoker_hypothesis_test_rejects_h0(clean_df: pd.DataFrame):
    results = smoker_hypothesis_test(clean_df)
    assert results["t_pval"] < 0.05
    assert results["u_pval"] < 0.05
    assert results["cohens_d"] > 0.8  # should be large effect


def test_plot_correlation_returns_figure(clean_df: pd.DataFrame):
    fig = plot_correlation_heatmap(clean_df, method="pearson")
    assert fig is not None


def test_plot_smoker_scatter_returns_figure(clean_df: pd.DataFrame):
    fig = plot_smoker_scatter(clean_df)
    assert fig is not None


def test_analyze_target_runs(clean_df: pd.DataFrame, capsys):
    analyze_target(clean_df)
    captured = capsys.readouterr()
    assert "smoker=yes" in captured.out
    assert "smoker=no" in captured.out


def test_analyze_interactions_runs(clean_df: pd.DataFrame, capsys):
    analyze_interactions(clean_df)
    captured = capsys.readouterr()
    assert "BMI" in captured.out


def test_compute_vif_returns_dataframe(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    X_train_eng, _ = add_interaction_features(X_train_proc, X_test_proc)
    vif_df = compute_vif(X_train_eng)
    assert "VIF" in vif_df.columns
    assert "feature" in vif_df.columns
    assert len(vif_df) == X_train_eng.shape[1]
    assert (vif_df["VIF"] > 0).all()
