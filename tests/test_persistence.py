import pandas as pd
from sklearn.linear_model import LinearRegression

from src.persistence import FullPipeline, PredictionPipeline, load_model, save_model


def test_full_pipeline_fit_transform(clean_df: pd.DataFrame):
    X = clean_df.drop(columns=["charges"])
    pipeline = FullPipeline()
    pipeline.fit(X)
    result = pipeline.transform(X)
    # 9 base + 4 interaction = 13
    assert result.shape[1] == 13
    assert result.shape[0] == len(X)


def test_prediction_pipeline_roundtrip(clean_df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    X = clean_df.drop(columns=["charges"])
    y = clean_df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    full_pipeline = FullPipeline()
    model = LinearRegression()

    pred_pipeline = PredictionPipeline(model, full_pipeline)
    pred_pipeline.fit(X_train, y_train)

    # Predict on held-out test data (not training data)
    preds = pred_pipeline.predict(X_test.head(5))
    assert len(preds) == 5
    assert all(p > 0 for p in preds)
    # Predictions should be in a reasonable range for insurance costs
    assert all(500 < p < 80000 for p in preds)


def test_save_and_load_model(clean_df: pd.DataFrame, tmp_path, monkeypatch):
    monkeypatch.setattr("src.persistence.MODEL_DIR", tmp_path)

    X = clean_df.drop(columns=["charges"])
    y = clean_df["charges"]

    full_pipeline = FullPipeline()
    full_pipeline.fit(X)
    model = LinearRegression()
    X_transformed = full_pipeline.transform(X)
    model.fit(X_transformed, y)

    save_model(model, full_pipeline, "test_model")
    assert (tmp_path / "test_model.joblib").exists()

    monkeypatch.setattr("src.persistence.MODEL_DIR", tmp_path)
    loaded = load_model("test_model")
    loaded_preds = loaded.predict(X.head(3))
    assert len(loaded_preds) == 3
