import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import (
    CATEGORICAL_BINARY,
    CATEGORICAL_NOMINAL,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    SEX_ORDER,
    SMOKER_ORDER,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified train/test split."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=X[["smoker", "sex", "region"]],
    )
    logger.info(
        "data.split train_rows=%d test_rows=%d ratio=%.0f/%.0f stratify=smoker,sex,region",
        X_train.shape[0],
        X_test.shape[0],
        (1 - TEST_SIZE) * 100,
        TEST_SIZE * 100,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """sklearn ColumnTransformer for all feature types."""
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), NUMERICAL_FEATURES),
            (
                "bin",
                OrdinalEncoder(categories=[SEX_ORDER, SMOKER_ORDER], dtype=np.int8),
                CATEGORICAL_BINARY,
            ),
            ("nom", OneHotEncoder(sparse_output=False, dtype=np.int8), CATEGORICAL_NOMINAL),
        ]
    )


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit on train, transform both splits, return DataFrames with column names."""
    if preprocessor is None:
        preprocessor = build_preprocessor()

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    ohe_names = (
        preprocessor.named_transformers_["nom"].get_feature_names_out(CATEGORICAL_NOMINAL).tolist()
    )
    columns = NUMERICAL_FEATURES + CATEGORICAL_BINARY + ohe_names

    X_train_df = pd.DataFrame(X_train_arr, columns=columns).astype(np.float32)
    X_test_df = pd.DataFrame(X_test_arr, columns=columns).astype(np.float32)

    logger.info(
        "features.encoded total=%d numerical=%d binary=%d onehot=%d",
        len(columns),
        len(NUMERICAL_FEATURES),
        len(CATEGORICAL_BINARY),
        len(ohe_names),
    )
    return X_train_df, X_test_df, preprocessor


def _add_interactions(df: pd.DataFrame, bmi_threshold: float) -> None:
    """Add interaction columns to a single DataFrame (mutates in place)."""
    df["smoker_x_bmi"] = df["smoker"] * df["bmi"]
    df["smoker_x_age"] = df["smoker"] * df["age"]
    df["age_sq"] = df["age"] ** 2
    df["obese_smoker"] = ((df["bmi"] > bmi_threshold) & (df["smoker"] == 1)).astype(np.float32)


def add_interaction_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create interaction and polynomial features. Thresholds from training data only."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    bmi_threshold = X_train["bmi"].quantile(0.55)
    _add_interactions(X_train, bmi_threshold)
    _add_interactions(X_test, bmi_threshold)

    logger.info(
        "features.interactions added=4 bmi_threshold=%.3f total_features=%d",
        bmi_threshold,
        X_train.shape[1],
    )
    return X_train, X_test
