import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import MODEL_DIR
from src.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


class FullPipeline(BaseEstimator, TransformerMixin):
    """End-to-end pipeline: raw DataFrame -> preprocessed features with interactions."""

    def __init__(self):
        self.preprocessor = build_preprocessor()
        self.bmi_threshold = None
        self.columns_base = None

    def fit(self, X, y=None):
        from src.config import CATEGORICAL_BINARY, NUMERICAL_FEATURES

        X_arr = self.preprocessor.fit_transform(X)

        ohe_names = (
            self.preprocessor.named_transformers_["nom"].get_feature_names_out(["region"]).tolist()
        )
        self.columns_base = NUMERICAL_FEATURES + CATEGORICAL_BINARY + ohe_names

        X_df = pd.DataFrame(X_arr, columns=self.columns_base).astype(np.float32)
        self.bmi_threshold = X_df["bmi"].quantile(0.55)
        return self

    def transform(self, X):
        X_arr = self.preprocessor.transform(X)
        X_df = pd.DataFrame(X_arr, columns=self.columns_base).astype(np.float32)

        X_df["smoker_x_bmi"] = X_df["smoker"] * X_df["bmi"]
        X_df["smoker_x_age"] = X_df["smoker"] * X_df["age"]
        X_df["age_sq"] = X_df["age"] ** 2
        X_df["obese_smoker"] = ((X_df["bmi"] > self.bmi_threshold) & (X_df["smoker"] == 1)).astype(
            np.float32
        )

        return X_df


class PredictionPipeline(BaseEstimator):
    """Complete prediction pipeline: raw data -> predicted charges."""

    def __init__(self, model, full_pipeline):
        self.model = model
        self.full_pipeline = full_pipeline

    def fit(self, X, y):
        self.full_pipeline.fit(X)
        X_transformed = self.full_pipeline.transform(X)
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.full_pipeline.transform(X)
        return self.model.predict(X_transformed)


def save_model(model, full_pipeline, name: str = "best_model") -> Path:
    """Save model + preprocessing pipeline to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    pipeline = PredictionPipeline(model, full_pipeline)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("model.saved path=%s size_kb=%.1f", path, path.stat().st_size / 1024)
    return path


def load_model(name: str = "best_model") -> PredictionPipeline:
    """Load model pipeline from disk."""
    path = MODEL_DIR / f"{name}.joblib"
    logger.info("model.loaded path=%s", path)
    return joblib.load(path)
