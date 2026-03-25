import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH

logger = logging.getLogger(__name__)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the insurance dataset and drop duplicate rows."""
    df = pd.read_csv(path)
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.info("duplicates.removed count=%d strategy=keep_first", n_dupes)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info("dataset.loaded rows=%d cols=%d source=%s", len(df), len(df.columns), path.name)
    return df
