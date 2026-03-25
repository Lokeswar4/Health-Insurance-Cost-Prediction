from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "insurance.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTUNA_TRIALS = 30

NUMERICAL_FEATURES = ["bmi", "children", "age"]
CATEGORICAL_BINARY = ["sex", "smoker"]
CATEGORICAL_NOMINAL = ["region"]
TARGET = "charges"

SEX_ORDER = ["female", "male"]
SMOKER_ORDER = ["no", "yes"]
