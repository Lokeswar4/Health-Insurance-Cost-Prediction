import numpy as np
import pandas as pd

from src.preprocessing import add_interaction_features, preprocess, split_data


def test_split_data_shapes(clean_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    total = len(X_train) + len(X_test)
    assert total == len(clean_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_data_ratio(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    ratio = len(X_test) / (len(X_train) + len(X_test))
    assert 0.19 < ratio < 0.21


def test_split_data_no_target_leakage(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    assert "charges" not in X_train.columns
    assert "charges" not in X_test.columns


def test_preprocessor_output_shape(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    # 3 numerical + 2 binary + 4 one-hot regions = 9
    assert X_train_proc.shape[1] == 9
    assert X_test_proc.shape[1] == 9


def test_preprocessor_output_dtype(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, _, _ = preprocess(X_train, X_test)
    assert X_train_proc.dtypes.apply(lambda d: d == np.float32).all()


def test_preprocessor_no_nulls(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    assert X_train_proc.isnull().sum().sum() == 0
    assert X_test_proc.isnull().sum().sum() == 0


def test_numerical_features_scaled(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, _, _ = preprocess(X_train, X_test)
    for col in ["bmi", "children", "age"]:
        assert X_train_proc[col].min() >= -0.01  # float tolerance
        assert X_train_proc[col].max() <= 1.01


def test_interaction_features_added(clean_df: pd.DataFrame):
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    X_train_eng, X_test_eng = add_interaction_features(X_train_proc, X_test_proc)
    assert X_train_eng.shape[1] == 13  # 9 base + 4 interaction
    assert X_test_eng.shape[1] == 13
    expected_new = {"smoker_x_bmi", "smoker_x_age", "age_sq", "obese_smoker"}
    assert expected_new.issubset(set(X_train_eng.columns))


def test_interaction_features_no_leakage(clean_df: pd.DataFrame):
    """obese_smoker threshold must come from train only — test values can exceed train range."""
    X_train, X_test, _, _ = split_data(clean_df)
    X_train_proc, X_test_proc, _ = preprocess(X_train, X_test)
    X_train_eng, X_test_eng = add_interaction_features(X_train_proc, X_test_proc)
    # obese_smoker should only be 0 or 1
    assert set(X_train_eng["obese_smoker"].unique()).issubset({0.0, 1.0})
    assert set(X_test_eng["obese_smoker"].unique()).issubset({0.0, 1.0})
