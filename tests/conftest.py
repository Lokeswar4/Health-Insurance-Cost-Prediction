import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.config import DATA_PATH


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Load the raw insurance dataset (with duplicates)."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated dataset matching data_loader.load_data output."""
    return raw_df.drop_duplicates(keep="first").reset_index(drop=True)
