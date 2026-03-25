from src.config import DATA_PATH
from src.data_loader import load_data


def test_load_data_returns_expected_columns():
    df = load_data()
    expected = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    assert set(df.columns) == expected


def test_load_data_drops_duplicates():
    df = load_data()
    assert df.duplicated().sum() == 0


def test_load_data_not_empty():
    df = load_data()
    assert len(df) > 1000


def test_load_data_with_explicit_path():
    df = load_data(path=DATA_PATH)
    assert len(df) > 0
